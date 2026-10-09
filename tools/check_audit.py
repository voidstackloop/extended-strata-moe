"""tools/check_audit.py - checks llama.cpp's --moe-cache-audit records against what they claim.

llama-completion writes a load and an exit record; llama-server also a ready record once loaded and one record per
request.  Checked: the profile's SHA-256 is the file's, the experts held per layer add up to the prefilled slots (no
duplicates, no more than the slots), the per-layer counters add up to the totals, the hybrid never uploads or changes
what it holds, and the server's per-request counters add up to exit - ready (one slot).

    python tools/check_audit.py MODEL.gguf PROFILE.strp [--cache-mib 256] [--bin DIR]      (DIR: tools/llama_env.py)
"""
import argparse
import hashlib
import json
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


def check_load(rec, profile, hybrid):
    assert rec["event"] == "load", rec["event"]
    assert rec["hybrid"] == hybrid
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
    held = check_load(recs[0], a.profile, hybrid)
    ex = recs[1]
    assert ex["hits"] + ex["misses"] > 0, "no lookups counted"
    if hybrid:   # the hybrid never uploads and never changes what it holds
        assert ex["bytes_upload"] == 0 and ex["hits"] == ex["hits_profile"]
        assert [l["resident"] for l in ex["layers"]] == [l["resident"] for l in recs[0]["layers"]]
    assert sum(l["hits"] for l in ex["layers"]) == ex["hits"] and sum(l["misses"] for l in ex["layers"]) == ex["misses"]
    print(f"completion{' (hybrid)' if hybrid else ''}: ok - {held} experts held, "
          f"{ex['hits']} hits, {ex['misses']} misses, {ex['bytes_cpu'] / 2**20:.1f} MiB CPU, {ex['bytes_upload'] / 2**20:.1f} MiB uploaded")


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
    check_load(recs[0], a.profile, True)
    ready, reqs, ex = recs[1], recs[2:4], recs[4]
    # the server's own decodes at load (a context probe; no warmup here) are between load and ready
    diff = {c: (sum(r[f"req_{c}"] for r in reqs), ex[c] - ready[c]) for c in ("hits", "misses", "bytes_cpu")}
    assert all(got == want for got, want in diff.values()), \
        f"requests' sums vs exit - ready: {diff}; per request (n_prompt, n_gen): {[(r['n_prompt'], r['n_gen']) for r in reqs]}"
    for r in reqs:
        assert r["n_gen"] > 0 and r["req_hits"] + r["req_misses"] > 0
    print(f"server: ok - {len(reqs)} request records add up to exit - ready ({ex['hits'] - ready['hits']} hits, "
          f"{ex['misses'] - ready['misses']} misses; {ready['hits'] + ready['misses']} lookups at load)")


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
    check_server(a, env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
