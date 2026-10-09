"""tools/arch_matrix.py - the tracer and the planner on every MoE architecture llama.cpp has, CPU only.

The models are llama.cpp's own tiny random ones: `test-llama-archs -a . -o DIR` writes `<arch>-moe.gguf` for each.
Per model: the planner reads it, the tracer routes a text through it, and every layer the planner says has experts
must be traced for every token.  Writes a markdown table; exit 1 if any architecture fails.

    python tools/arch_matrix.py DIR [--out matrix.md] [--trace-bin build/routing_trace]
"""
import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import moe_plan as mp

ROOT = Path(__file__).resolve().parent.parent

# fixtures that test-llama-archs saves as "-moe" but whose graph never routes: checked by hand, not by this script
NOT_ROUTED = {
    "refact": "n/a: loads expert tensors, but its graph runs the dense FFN (src/models/refact.cpp)",
}


def check(model, tracer, text, tmp):
    info = mp.inspect([model])
    expert_layers = sorted(l for g in info.groups for l in g.layers)
    groups = len(info.groups)
    if not expert_layers:
        return info, expert_layers, groups, None, "n/a: no expert tensors (a dense model)"
    if info.arch in NOT_ROUTED:
        return info, expert_layers, groups, None, NOT_ROUTED[info.arch]
    trace = tmp / (model.stem + ".bin")
    r = subprocess.run([tracer, model, text, trace, "128"], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or ["?"])[-1][:120]
        return info, expert_layers, groups, None, f"tracer failed: {tail}"
    per_layer = defaultdict(int)
    for (l, _), n in mp.read_traces([trace], info.n_layers, info.n_expert).items():
        per_layer[l] += n
    traced = sorted(per_layer)
    if traced != expert_layers:
        return info, expert_layers, groups, per_layer, f"traced layers {traced} != expert layers {expert_layers}"
    if len(set(per_layer.values())) != 1:
        return info, expert_layers, groups, per_layer, f"uneven routings per layer {dict(per_layer)}"
    # every token, not just the same number in every layer: a dropped ubatch drops it in all of them
    m = re.search(r"(\d+) tokens, \d+ routing records", r.stderr)
    want = int(m.group(1)) * info.n_used if m else None
    if want is None or next(iter(per_layer.values())) != want:
        return info, expert_layers, groups, per_layer, f"{next(iter(per_layer.values()))} routings per layer, expected {want}"
    return info, expert_layers, groups, per_layer, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out")
    ap.add_argument("--trace-bin", default=str(ROOT / "build" / "routing_trace"))
    a = ap.parse_args()
    d = Path(a.dir)
    tmp = d / "traces"
    tmp.mkdir(exist_ok=True)
    text = tmp / "text.txt"
    text.write_text(" ".join(f"word{i % 97} token{i % 13}." for i in range(800)))

    rows, failed = [], 0
    for model in sorted(d.glob("*-moe.gguf")):
        try:
            info, layers, groups, per_layer, status = check(model, a.trace_bin, text, tmp)
            used = next(iter(per_layer.values())) // info.n_used if per_layer else 0
            rows.append(f"| {info.arch} | {info.n_expert} / top-{info.n_used} | {len(layers)} of {info.n_layers} | "
                        f"{groups} | {used} | {status} |")
        except (Exception, SystemExit) as e:
            status = f"planner: {e}"[:140]
            rows.append(f"| {model.stem[:-4]} | | | | | {status} |")
        if not (rows[-1].endswith("| ok |") or "| n/a: " in rows[-1]):
            failed += 1
        print(rows[-1], flush=True)
    table = ["| arch | experts | layers with experts | cache groups | tokens traced per layer | result |",
             "|---|---|---|---|---|---|"] + rows
    if a.out:
        Path(a.out).write_text("\n".join(table) + "\n")
    na = sum("| n/a: " in r for r in rows)
    print(f"{len(rows) - failed - na} ok, {na} n/a, {failed} failed, of {len(rows)} architectures")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
