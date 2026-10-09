#!/bin/bash
# tools/e2e_tiny.sh - trace -> plan -> profile -> llama.cpp's reader, on a tiny random qwen3moe from llama.cpp's
# test-llama-archs (no download, CPU only).  Needs build/ (cmake -S tools/routing_trace -B build) and a
# test-llama-archs of the strata-moe build: $LLAMA_BIN, else what tools/llama_env.py finds.
set -e
cd "$(dirname "$0")/.."
D=build/tiny
mkdir -p $D
M=$D/qwen3moe-moe.gguf
[ -f $M ] || .venv/bin/python -I - $D <<'EOF'
import subprocess, sys
sys.path.insert(0, "tools")
import llama_env
b = llama_env.default_bin()
subprocess.run([f"{b}/test-llama-archs", "-a", "^qwen3moe$", "-o", sys.argv[1]], env=llama_env.env(b), capture_output=True, check=True)
EOF
cat README.md docs/*.md | head -c 20000 > $D/text.txt
./build/routing_trace $M $D/text.txt $D/trace.bin 256 2>&1 | tail -1
.venv/bin/python -I - "$M" "$D/trace.bin" <<'EOF'
import sys
sys.path.insert(0, "tools")
import moe_plan as m
i = m.inspect([sys.argv[1]])
f = m.read_traces([sys.argv[2]], i.n_layers, i.n_expert)
per_layer = {l: sum(f.get((l, e), 0) for e in range(i.n_expert)) for l in sorted(i.group_of())}
print("routings per layer:", per_layer)
assert len(set(per_layer.values())) == 1, "every expert layer must see every token"
EOF
.venv/bin/python -I tools/moe_plan.py $M --cache-mib 8 --trace $D/trace.bin --out $D/p.strp
NL=$(.venv/bin/python -I -c "from gguf import GGUFReader as R; r=R('$M'); a=r.get_field('general.architecture').contents(); print(r.get_field(a+'.block_count').contents(), r.get_field(a+'.expert_count').contents())")
[ "$(./build/read_profile $D/p.strp $NL | wc -l)" -gt 0 ]
echo "e2e_tiny: ok"
