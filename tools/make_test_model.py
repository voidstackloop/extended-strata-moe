"""tools/make_test_model.py - a mid-size MoE GGUF from one of test-llama-archs' tiny ones, for checks that need real
expert sizes (the tiny models' experts all fit in 1 MiB, the smallest --moe-cache-mib).

Copies the template's metadata and non-expert tensors, sets the expert count, and writes random experts: layer 0 in
the template's type (F32) and every other layer in Q8_0, so the MoE cache has two groups with different expert sizes.

    python tools/make_test_model.py TEMPLATE.gguf OUT.gguf [--n-expert 64] [--n-used 8]
"""
import argparse
import re

import numpy as np
from gguf import GGMLQuantizationType as Q, GGUFReader, GGUFValueType, GGUFWriter
from gguf.quants import quantize

EXPS_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
PER_EXPERT_RE = re.compile(r"^blk\.\d+\.(ffn_gate_inp\.weight|ffn_(gate|up|down)_exps\.(scale|input_scale))$")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("template")
    ap.add_argument("out")
    ap.add_argument("--n-expert", type=int, default=64)
    ap.add_argument("--n-used", type=int, default=8)
    a = ap.parse_args(argv)

    r = GGUFReader(a.template)
    arch = r.get_field("general.architecture").contents()
    w = GGUFWriter(a.out, arch)
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        val = f.contents()
        if name == f"{arch}.expert_count":
            val = a.n_expert
        elif name == f"{arch}.expert_used_count":
            val = a.n_used
        vtype = f.types[0]
        if vtype == GGUFValueType.ARRAY and len(val) == 0:
            continue                                            # gguf-py cannot write one; absent reads the same
        w.add_key_value(name, val, vtype, sub_type=f.types[-1] if vtype == GGUFValueType.ARRAY and len(f.types) > 1 else None)

    rng = np.random.default_rng(0)
    for t in r.tensors:
        shape = [int(x) for x in t.shape]                       # ne order
        m = EXPS_RE.match(t.name)
        if m:
            data = (rng.standard_normal((a.n_expert, shape[1], shape[0])) * 0.02).astype(np.float32)   # numpy order
            if int(m.group(1)) == 0:
                w.add_tensor(t.name, data)
            else:
                w.add_tensor(t.name, quantize(data, Q.Q8_0), raw_dtype=Q.Q8_0)
        elif PER_EXPERT_RE.match(t.name):
            if t.name.endswith("ffn_gate_inp.weight"):
                w.add_tensor(t.name, (rng.standard_normal((a.n_expert, shape[0])) * 0.1).astype(np.float32))
            else:
                w.add_tensor(t.name, np.ones(a.n_expert, np.float32))
        elif t.tensor_type in (Q.F32, Q.F16):
            w.add_tensor(t.name, np.array(t.data))
        else:
            w.add_tensor(t.name, np.array(t.data), raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out}: {arch}, {a.n_expert} experts, top-{a.n_used}")


if __name__ == "__main__":
    main()
