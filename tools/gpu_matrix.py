"""tools/gpu_matrix.py - llama.cpp's own correctness test, per MoE architecture, with the experts in RAM and the MoE cache.

Runs `test-llama-archs -a ^ARCH$ --all-host-experts -b CUDA0` (the extended-strata-moe branch) once per architecture, so one
crash cannot hide the others.  Each run compares with the CPU's logits (NMSE <= 1e-4, a save/load round trip, a mixed
batch) these configs: the GPU alone, the experts in RAM ("Host experts"), the MoE cache, the cache filled from a
profile, a cache that holds every expert from a profile (nothing uploaded after the load: every expert the GPU
reads is one the prefill wrote), and the adaptive hybrid cache (experts swapped in between chunks of tokens; the
config fails if nothing was swapped).

With --quant-experts the routed experts are Q8_0, so the one-token hybrid config takes CUDA's skip path, as in
generation.

Verdicts: "ok" - every config passes.  "ok, upstream" - a config fails, but no cache config fails a check that "Host
experts" (the same placement without the cache) passes, or fails it by more than 5% NMSE: the difference is
llama.cpp's GPU against its CPU, not the cache.  A cache-only check (bit identity, the first MoE layer) never passes
this way.  "n/a" - the fixture has no expert tensors.  "FAIL" - anything else.

    python tools/gpu_matrix.py ARCH_DIR [--bin DIR] [--out gpu_matrix.md] [--quant-experts]   (DIR: tools/llama_env.py)
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

from gguf import GGUFReader

sys.path.insert(0, str(Path(__file__).parent))
import llama_env
import moe_plan as mp

N_CACHE_CONFIGS = 8   # MoE cache; + profile; + full profile; hybrid, half; hybrid, full; full, 1 token; hybrid, 1 token; adaptive
ROW_RE = re.compile(r"^\|\s*(\S+)\s*\|\s*([^|]+?)\s*\|\s*(\S+)\s*\|(.*)$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# "(1.2e-05)", or a cache-only check that failed: "(3.0e-07 c)" logits not bit-identical to the cache without a
# profile, "(2.1e-02 m)" the hybrid's first MoE layer off the cache holding every expert, "(1.0e-07 s)" the adaptive
# cache swapped nothing
NMSE_RE = re.compile(r"\(([0-9.e+-]+|nan|inf)(?: ([cms]))?\)")


def cols(rest):
    """(nmse status, nmse value, cache-only check marker, roundtrip status, mixed status) of one result row."""
    c = [x.strip() for x in rest.split("|")[:3]]
    m = NMSE_RE.search(c[0])
    return (c[0].split()[0] if c[0] else "", float(m.group(1)) if m else None, m.group(2) if m else None,
            c[1].split()[0] if c[1] else "", c[2].split()[0] if c[2] else "")


def no_worse(c, base):
    """The cache config fails no check the no-cache config passes, and where both fail the NMSE, it is not >5% worse."""
    if any(c[i] == "FAIL" and base[i] != "FAIL" for i in (0, 3, 4)):
        return False
    return not (c[0] == "FAIL" and c[1] is not None and base[1] is not None and c[1] > 1.05 * base[1])


def verdict(out, rc):
    rows = [m.groups() for m in map(ROW_RE.match, out.splitlines()) if m and m.group(3) == "MoE"]
    configs = {dev: cols(rest) for _, dev, _, rest in rows}
    # the NMSE column says SKIP when the test did not run a config: that is not a pass; NaN must not pass either
    cache = {d: c for d, c in configs.items() if d.startswith(("MoE cache", "MoE hybrid")) and c[0] != "SKIP"}
    if rc not in (0, 1) or len(cache) < N_CACHE_CONFIGS:   # 1 is "a config failed", judged below; anything else crashed
        tail = [ln.strip()[:110] for ln in out.splitlines() if "error" in ln.lower() or "GGML_ASSERT" in ln][-1:]
        return f"FAIL (rc={rc}, {len(cache)} of {N_CACHE_CONFIGS} cache configs ran) {' '.join(tail)}"
    nan = [d for d, c in configs.items() if c[1] is not None and c[1] != c[1]]
    if nan:
        return f"FAIL: NaN in {', '.join(nan)}"
    own = [f"{d} ({c[2]})" for d, c in cache.items() if c[0] == "FAIL" and c[2]]
    if own:   # a cache-only check: nothing upstream can explain it
        return f"FAIL: {', '.join(own)}"
    failing = [d for d, c in configs.items() if "FAIL" in (c[0], c[3], c[4])]
    if not failing:
        return "ok"
    if not any(d in failing for d in cache):
        return f"ok, upstream: {', '.join(failing)} fail without the cache; every cache config passes"
    base = configs.get("Host experts")
    if base and all(no_worse(c, base) for c in cache.values()):
        what = ", ".join(f"{name} {s}" for name, s in (("NMSE", base[0]), ("roundtrip", base[3]), ("mixed batch", base[4]))
                         if s == "FAIL")
        nmse = f" {base[1]:.2e}" if base[1] is not None and base[0] == "FAIL" else ""
        return f"ok, upstream: GPU vs CPU {what}{nmse} without the cache too; no cache config does worse"
    return f"FAIL: {', '.join(failing)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--bin", default=llama_env.default_bin())
    ap.add_argument("--out")
    ap.add_argument("--timeout", type=int, default=300, help="seconds per architecture before it counts as failed")
    ap.add_argument("--quant-experts", action="store_true",
                    help="Q8_0 routed experts: the hybrid's one-token config then takes CUDA's skip path, as in generation")
    a = ap.parse_args()
    env = llama_env.env(a.bin)
    models = {GGUFReader(p).get_field("general.architecture").contents(): p for p in Path(a.dir).glob("*-moe.gguf")}

    lines, bad = [], []
    for arch in sorted(models):
        if not mp.inspect([models[arch]]).groups:
            v = "n/a: no expert tensors (a dense model)"
        else:
            try:   # a fixture takes seconds; a hang is a failure, reported at once, not a crash after half an hour
                r = subprocess.run([str(Path(a.bin) / "test-llama-archs"), "-a", f"^{re.escape(arch)}$", "--all-host-experts",
                                    "-b", "CUDA0", "-s", "42"] + (["--quant-experts"] if a.quant_experts else []),
                                   capture_output=True, text=True, env=env, timeout=a.timeout)
                v = verdict(ANSI_RE.sub("", r.stdout + r.stderr), r.returncode)
            except subprocess.TimeoutExpired:
                v = f"FAIL: no result in {a.timeout} s"
        if v.startswith("FAIL"):
            bad.append(arch)
        lines.append(f"| {arch} | {v} |")
        print(lines[-1], flush=True)
    table = ["| arch | result |", "|---|---|"] + lines
    if a.out:
        Path(a.out).write_text("\n".join(table) + "\n")
    print(f"{len(models) - len(bad)} of {len(models)} architectures ok or n/a" + (f"; failed: {', '.join(bad)}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
