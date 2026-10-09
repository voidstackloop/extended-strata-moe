"""tools/bench_cache.py - decode speed and hit rate of the expert placements, measured with llama-completion.

Each config runs the same prompt (the start of TEXT) and greedy generation; reported: generation tokens/s (llama.cpp's
own eval timing) and the MoE cache's hit counter for single-token ubatches. The runs are interleaved (round r of every
config, then round r+1), because a laptop's speed drifts with its temperature by more than the configs differ.

    python tools/bench_cache.py MODEL.gguf TEXT --profile P.strp --cache-mib 64 256 [--n-cpu-moe 22 16] [-t 4 6] [-n 128] [--runs 2]

Compare each config at its best thread count: on gpt-oss-20b the layer split was best at 6 threads, the hybrid at 4-6.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llama_env

EVAL_RE = re.compile(r"\beval time = +([\d.]+) ms / +(\d+) runs .*?([\d.]+) tokens per second")
STATS_RE = re.compile(r"ubatch <= 8: hits = (\d+)(?: \((\d+) from the profile[^)]*\))?, misses = (\d+), hit rate = ([\d.]+)%, uploaded = ([\d.]+) MiB")
HYBRID_RE = re.compile(r"ubatch <= 8: hybrid: hits = (\d+), computed on the CPU = (\d+), hit rate = ([\d.]+)%, CPU read ([\d.]+) MiB")


def run(binary, model, prompt, n, extra):
    cmd = [binary, "-m", model, "-p", prompt, "-n", str(n), "-no-cnv", "--no-warmup", "-s", "1", "--temp", "0",
           "-ngl", "999", "-fit", "off", "-v"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
    out = r.stdout + r.stderr
    ev = [m for m in EVAL_RE.finditer(out)]
    st = STATS_RE.search(out)
    hy = HYBRID_RE.search(out)
    if r.returncode != 0 or not ev:
        return None, out
    tps = float(ev[-1].group(3))
    if hy:   # (hits, from profile, misses, rate, uploaded): the hybrid's misses are read by the CPU, not uploaded
        hits, cpu, rate, mib = hy.groups()
        return (tps, (hits, hits, cpu, rate, f"0 (CPU read {mib} MiB)")), out
    if st:
        hits, prof, misses, rate, up = st.groups()
        return (tps, (hits, prof, misses, rate, f"{up} MiB")), out
    return (tps, None), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("text")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--cache-mib", type=int, nargs="*", default=[])
    ap.add_argument("--n-cpu-moe", type=int, nargs="*", default=[])
    ap.add_argument("--hybrid", action="store_true", help="also run the hybrid cache (--moe-cache-hybrid)")
    ap.add_argument("--only-hybrid", action="store_true", help="of the cache configs, run only the hybrid")
    ap.add_argument("-t", "--threads", type=int, nargs="*", default=[],
                    help="CPU thread counts to run every config at (default: llama.cpp's own)")
    ap.add_argument("-n", type=int, default=128)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--prompt-chars", type=int, default=1500)
    ap.add_argument("--bin", default=llama_env.default_bin())
    a = ap.parse_args()
    os.environ.update(llama_env.env(a.bin))
    binary = str(Path(a.bin) / "llama-completion")
    prompt = Path(a.text).read_text(errors="replace")[:a.prompt_chars]

    configs = [("all on GPU", []), ("--cpu-moe (no cache)", ["--cpu-moe"])]
    configs += [(f"--n-cpu-moe {n}", ["--n-cpu-moe", str(n)]) for n in a.n_cpu_moe]
    for mib in a.cache_mib:
        base = ["--cpu-moe", "--moe-cache-mib", str(mib)]
        if not a.only_hybrid:
            configs += [(f"cache {mib} MiB, cold", base), (f"cache {mib} MiB, profile", base + ["--moe-cache-profile", a.profile])]
        if a.hybrid or a.only_hybrid:
            configs.append((f"hybrid {mib} MiB", base + ["--moe-cache-profile", a.profile, "--moe-cache-hybrid"]))
    if a.threads:
        configs = [(f"{name}, -t {t}", extra + ["-t", str(t)]) for name, extra in configs for t in a.threads]

    best = {}
    failed = {}
    for _ in range(a.runs):
        for name, extra in configs:
            if name in failed:
                continue
            res, out = run(binary, a.model, prompt, a.n, extra)
            if res is None:
                lines = out.strip().splitlines()
                failed[name] = lines[-1][:100] if lines else "no output"
            elif name not in best or res[0] > best[name][0]:
                best[name] = res

    print(f"| config | gen tokens/s (best of {a.runs}, interleaved) | decode hit rate | from profile | uploaded |")
    print("|---|---:|---:|---:|---:|")
    for name, _ in configs:
        if name in failed:
            print(f"| {name} | failed: {failed[name]} | | | |")
            continue
        tps, st = best[name]
        if st:
            hits, prof, misses, rate, up = st
            print(f"| {name} | {tps:.1f} | {rate}% | {prof or '-'} of {hits} | {up} |")
        else:
            print(f"| {name} | {tps:.1f} | | | |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
