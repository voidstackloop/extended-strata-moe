"""tools/check_slots.py - the planner's cache arithmetic against llama.cpp's own, on a GPU.

For a range of --moe-cache-mib sizes, runs llama-completion (the extended-strata-moe branch, CUDA) on MODEL with the experts
in RAM and compares what llama.cpp logs - the slots of every cache group, a refused cache below the minimum, the
experts a profile prefills - with what tools/moe_plan.py predicts.  Exit 1 on any difference.

    python tools/check_slots.py MODEL.gguf [--bin DIR]      (default DIR: see tools/llama_env.py)
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llama_env
import moe_plan as mp

SLOTS_RE = re.compile(r"impl: +(\d+) layers, (\S+): +(\d+) slots")      # llama_moe_cache's constructor, -v only
FILLED_RE = re.compile(r"MoE cache filled with (\d+) experts")
STATS_RE = re.compile(r"ubatch <= 8: hits = (\d+), misses = (\d+)")


def run(binary, model, mib, profile=None):
    cmd = [binary, "-m", model, "-p", "The quick brown fox jumps over the lazy dog", "-n", "8", "-no-cnv", "--no-warmup",
           "-s", "1", "-ngl", "999", "-fit", "off", "--cpu-moe", "--moe-cache-mib", str(mib), "-v"]
    if profile:
        cmd += ["--moe-cache-profile", profile]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--bin", default=llama_env.default_bin())
    a = ap.parse_args()
    binary = str(Path(a.bin) / "llama-completion")
    os.environ.update(llama_env.env(a.bin))
    info = mp.inspect([a.model])
    need = mp.min_cache_mib(info)
    total_mib = info.host_bytes // mp.MIB + 1
    failures = 0

    rc, out = run(binary, a.model, need - 1)
    ok = rc != 0 and "too small to hold the experts of one token" in out
    print(f"{need - 1:5d} MiB: planner says refused, llama.cpp {'refused' if rc else 'ran'} -> {'ok' if ok else 'DIFFERENT'}")
    failures += not ok

    profile = Path(tempfile.mkdtemp()) / "p.strp"
    for mib in sorted({need, need + 1, need * 2, total_mib // 2, total_mib * 2}):
        mp.size_cache(info, mib * mp.MIB)
        want = [g.n_slots for g in info.groups]
        hot = mp.hot_set(info, mp.rank_pairs(info, {}))
        mp.write_profile(profile, info.n_layers, info.n_expert, mp.rank_pairs(info, {}), len(hot))
        rc, out = run(binary, a.model, mib, profile)
        got = [int(m.group(3)) for m in SLOTS_RE.finditer(out)]
        filled = [int(m.group(1)) for m in FILLED_RE.finditer(out)]
        stats = STATS_RE.search(out)
        ok = rc == 0 and got == want and filled == [len(hot)]
        hits = f", hits {stats.group(1)} / misses {stats.group(2)}" if stats else ""
        print(f"{mib:5d} MiB: slots planner {want} llama.cpp {got}, prefilled planner {len(hot)} llama.cpp {filled}{hits}"
              f" -> {'ok' if ok else 'DIFFERENT'}")
        if not ok:
            failures += 1
            if rc != 0:
                print("\n".join(out.strip().splitlines()[-8:]))
    print("check_slots: ok" if not failures else f"check_slots: {failures} difference(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
