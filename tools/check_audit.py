"""tools/check_audit.py - checks llama.cpp's --moe-cache-audit records against what they claim.

llama-completion writes a load and an exit record; llama-server (and llama-cli, which runs it in-process) also a ready
record once loaded and one record per request.  Checked: every record's time (ms, in order) and pid, the load record's load time and GPUs, the exit
record's throughput (n_gen / t_gen_ms), the profile's SHA-256 is the file's, the experts held per layer add up to the prefilled slots (no
duplicates, no more than the slots), the per-layer counters add up to the totals, the hybrid never uploads or changes
what it holds, the adaptive cache fills itself from nothing (swaps, hits, no profile), and the server's per-request
counters add up to exit - ready (one slot).

    python tools/check_audit.py MODEL.gguf PROFILE.strp [--cache-mib 256] [--bin DIR]      (DIR: tools/llama_env.py)
"""
import argparse
import hashlib
import json
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llama_env

COUNTERS = ("hits", "hits_profile", "misses", "bytes_upload", "bytes_cpu")


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def check_common(recs):
    """every record: a UTC time to the millisecond, in order, and the one process's id"""
    assert all(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", r["time"]) for r in recs), [r["time"] for r in recs]
    assert [r["time"] for r in recs] == sorted(r["time"] for r in recs), "times out of order"
    assert len({r["pid"] for r in recs}) == 1 and recs[0]["pid"] > 0, "pid"


def check_perf(ex):
    p = ex["perf"]
    assert p["n_gen"] > 0 and p["t_gen_ms"] > 0 and p["gen_tps"] > 0, p
    assert abs(p["gen_tps"] - 1e3 * p["n_gen"] / p["t_gen_ms"]) < 1e-6 * p["gen_tps"], p


def check_load(rec, profile, hybrid):
    assert rec["event"] == "load", rec["event"]
    assert rec["hybrid"] == hybrid
    assert rec["load_ms"] > 0, "load_ms"
    for dev in rec["devices"]:   # none on a CPU-only build
        assert dev["name"] and 0 < dev["mib_free"] <= dev["mib_total"], dev
    assert rec["profile"]["sha256"] == hashlib.sha256(Path(profile).read_bytes()).hexdigest(), "profile SHA-256"
    assert rec["profile"]["bytes"] == Path(profile).stat().st_size
    assert rec["params"]["moe_cache_hybrid"] == hybrid
    held = 0
    for layer in rec["layers"]:
        r = layer["resident"]
        assert len(r) == len(set(r)), f"layer {layer['il']}: an expert twice"
        assert all(0 <= e for e in r) and len(r) <= layer["slots"], f"layer {layer['il']}: {len(r)} held, {layer['slots']} slots"
        held += len(r)
    assert held == rec["cache"]["prefilled"], f"{held} experts held, {rec['cache']['prefilled']} prefilled"
    assert all(rec[c] == 0 for c in COUNTERS), "counters at load"
    return held


def check_completion(a, env, hybrid):
    with tempfile.TemporaryDirectory() as d:
        audit = Path(d) / "audit.jsonl"
        cmd = [str(Path(a.bin) / "llama-completion"), "-m", a.model, "-p", "The capital of France is", "-n", "16",
               "-no-cnv", "--no-warmup", "-s", "1", "--temp", "0", "-ngl", "999", "-fit", "off", "--cpu-moe",
               "--moe-cache-mib", str(a.cache_mib), "--moe-cache-profile", a.profile, "--moe-cache-audit", str(audit)]
        if hybrid:
            cmd.append("--moe-cache-hybrid")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL, timeout=600)
        assert r.returncode == 0, r.stderr[-500:]
        recs = records(audit)
    assert [x["event"] for x in recs] == ["load", "exit"], [x["event"] for x in recs]
    check_common(recs)
    held = check_load(recs[0], a.profile, hybrid)
    ex = recs[1]
    assert ex["hits"] + ex["misses"] > 0, "no lookups counted"
    check_perf(ex)
    assert ex["perf"]["n_gen"] <= 16 and ex["perf"]["n_prompt"] > 0, ex["perf"]
    if hybrid:   # the hybrid never uploads and never changes what it holds
        assert ex["bytes_upload"] == 0 and ex["hits"] == ex["hits_profile"]
        assert [l["resident"] for l in ex["layers"]] == [l["resident"] for l in recs[0]["layers"]]
    assert sum(l["hits"] for l in ex["layers"]) == ex["hits"] and sum(l["misses"] for l in ex["layers"]) == ex["misses"]
    print(f"completion{' (hybrid)' if hybrid else ''}: ok - {held} experts held, "
          f"{ex['hits']} hits, {ex['misses']} misses, {ex['bytes_cpu'] / 2**20:.1f} MiB CPU, {ex['bytes_upload'] / 2**20:.1f} MiB uploaded")


def check_mailbox(a, env):
    """one-token generation hands each layer over through the mailbox, and produces the text the scheduler's copies do"""
    cmd = [str(Path(a.bin) / "llama-completion"), "-m", a.model, "-p", "The capital of France is", "-n", "48", "-no-cnv",
           "--no-warmup", "-s", "1", "--temp", "0", "-ngl", "999", "-fit", "off", "--cpu-moe", "--moe-cache-mib",
           str(a.cache_mib), "--moe-cache-profile", a.profile, "--moe-cache-hybrid", "-v"]  # -v: the log line that says so
    runs = {}
    for name, extra_env in (("mailbox", {}), ("copies", {"LLAMA_MOE_CACHE_NO_MAILBOX": "1"})):
        r = subprocess.run(cmd, capture_output=True, text=True, env={**env, **extra_env}, stdin=subprocess.DEVNULL, timeout=600)
        assert r.returncode == 0, r.stderr[-500:]
        runs[name] = r
    assert "through a mailbox" in runs["mailbox"].stderr, "the mailbox was not used (a GPU build with CUDA and a profile is needed)"
    assert "through a mailbox" not in runs["copies"].stderr
    assert runs["mailbox"].stdout.strip() == runs["copies"].stdout.strip(), \
        f"the mailbox changed the text:\n{runs['mailbox'].stdout!r}\n{runs['copies'].stdout!r}"
    print(f"mailbox: ok - the same {len(runs['mailbox'].stdout.split())} words as with the scheduler's copies")


def check_adaptive(a, env):
    """no profile, --moe-cache-adaptive: the cache starts empty and fills itself while generating, and the audit says so"""
    with tempfile.TemporaryDirectory() as d:
        audit = Path(d) / "audit.jsonl"
        cmd = [str(Path(a.bin) / "llama-completion"), "-m", a.model, "-p", "The capital of France is", "-n", "160", "-no-cnv",
               "--no-warmup", "-s", "1", "--temp", "0", "-ngl", "999", "-fit", "off", "--cpu-moe", "--moe-cache-mib",
               str(a.cache_mib), "--moe-cache-hybrid", "--moe-cache-adaptive", "--moe-cache-audit", str(audit)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL, timeout=600)
        assert r.returncode == 0, r.stderr[-500:]
        recs = records(audit)
    assert [x["event"] for x in recs] == ["load", "exit"], [x["event"] for x in recs]
    check_common(recs)
    load, ex = recs
    assert load["params"]["moe_cache_adaptive"] and load["profile"] is None and load["cache"]["prefilled"] == 0, load["params"]
    assert load["swaps"] == 0 and all(l["resident"] == [] for l in load["layers"]), "not empty at load"
    held = 0
    for layer in ex["layers"]:
        res = layer["resident"]
        assert len(res) == len(set(res)) and len(res) <= layer["slots"], f"layer {layer['il']}: {len(res)} held, {layer['slots']} slots"
        held += len(res)
    assert ex["swaps"] >= held > 0, f"{ex['swaps']} swaps, {held} experts held"   # each held expert was swapped in
    assert ex["bytes_upload"] == 0 and ex["hits_profile"] == 0, "nothing comes from a profile or is uploaded"
    assert sum(l["hits"] for l in ex["layers"]) == ex["hits"] and sum(l["misses"] for l in ex["layers"]) == ex["misses"]
    rate = ex["hits"] / (ex["hits"] + ex["misses"])
    assert rate > 0.3, f"{rate:.0%} of the lookups hit: an empty hybrid cache hits 0%, the adaptive one should have learned"
    check_perf(ex)
    print(f"adaptive: ok - from an empty cache, {ex['swaps']} swaps filled {held} slots; {rate:.0%} of the lookups hit")


def check_server(a, env):
    port = 8097
    with tempfile.TemporaryDirectory() as d:
        audit = Path(d) / "audit.jsonl"
        cmd = [str(Path(a.bin) / "llama-server"), "-m", a.model, "--port", str(port), "-np", "1", "--no-warmup",
               "-ngl", "999", "-fit", "off", "--cpu-moe", "--moe-cache-mib", str(a.cache_mib), "--moe-cache-profile",
               a.profile, "--moe-cache-hybrid", "--moe-cache-audit", str(audit)]
        srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        try:
            for _ in range(240):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                    break
                except OSError:
                    time.sleep(0.5)
            for prompt in ("Paris is", "The sea is"):
                req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", method="POST",
                                             data=json.dumps({"prompt": prompt, "n_predict": 12, "temperature": 0}).encode(),
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=120).read()
        finally:
            srv.send_signal(signal.SIGINT)
            srv.wait(timeout=60)
        recs = records(audit)
    events = [x["event"] for x in recs]
    assert events == ["load", "ready", "request", "request", "exit"], events
    check_common(recs)
    check_load(recs[0], a.profile, True)
    ready, reqs, ex = recs[1], recs[2:4], recs[4]
    check_perf(ex)
    # the server's own decodes at load (a context probe; no warmup here) are between load and ready
    diff = {c: (sum(r[f"req_{c}"] for r in reqs), ex[c] - ready[c]) for c in ("hits", "misses", "bytes_cpu")}
    assert all(got == want for got, want in diff.values()), \
        f"requests' sums vs exit - ready: {diff}; per request (n_prompt, n_gen): {[(r['n_prompt'], r['n_gen']) for r in reqs]}"
    for r in reqs:
        assert r["n_gen"] > 0 and r["req_hits"] + r["req_misses"] > 0
    print(f"server: ok - {len(reqs)} request records add up to exit - ready ({ex['hits'] - ready['hits']} hits, "
          f"{ex['misses'] - ready['misses']} misses; {ready['hits'] + ready['misses']} lookups at load)")


def check_cli(a, env):
    """llama-cli runs the server in-process: the same records, one request for its one turn"""
    with tempfile.TemporaryDirectory() as d:
        audit = Path(d) / "audit.jsonl"
        cmd = [str(Path(a.bin) / "llama-cli"), "-m", a.model, "-p", "Paris is", "-n", "8", "-st", "-c", "4096",
               "-ngl", "999", "-fit", "off", "--cpu-moe", "--moe-cache-mib", str(a.cache_mib), "--moe-cache-profile",
               a.profile, "--moe-cache-hybrid", "--moe-cache-audit", str(audit)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL, timeout=600)
        assert r.returncode == 0, r.stderr[-500:]
        recs = records(audit)
    events = [x["event"] for x in recs]
    assert events == ["load", "ready", "request", "exit"], events
    check_common(recs)
    check_load(recs[0], a.profile, True)
    check_perf(recs[3])
    assert recs[2]["n_gen"] > 0
    print(f"cli: ok - {recs[2]['n_gen']} tokens, {recs[3]['perf']['gen_tps']:.1f} t/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("profile")
    ap.add_argument("--cache-mib", type=int, default=256)
    ap.add_argument("--bin", default=llama_env.default_bin())
    a = ap.parse_args()
    env = llama_env.env(a.bin)
    check_completion(a, env, hybrid=False)
    check_completion(a, env, hybrid=True)
    check_mailbox(a, env)
    check_adaptive(a, env)
    check_server(a, env)
    check_cli(a, env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
