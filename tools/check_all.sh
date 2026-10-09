#!/bin/bash
# tools/check_all.sh - every check, CPU then GPU.  Needs scripts/setup.sh run once (a CUDA build for the GPU checks).
#   bash tools/check_all.sh [--cpu-only]
set -e
cd "$(dirname "$0")/.."
PY=".venv/bin/python -I"
cmake --build build > /dev/null
if [ ! -f build/archs/qwen3moe-moe.gguf ]; then   # a tiny model of every architecture, from llama.cpp's own test
    mkdir -p build/archs
    $PY - <<'EOF'
import subprocess, sys
sys.path.insert(0, "tools")
import llama_env
b = llama_env.default_bin()
subprocess.run([f"{b}/test-llama-archs", "-a", ".", "-o", "build/archs"], env=llama_env.env(b), capture_output=True, check=True)
EOF
fi
echo "== planner unit tests";            $PY tools/test_moe_plan.py | tail -2
echo "== tracer end to end";             bash tools/e2e_tiny.sh | tail -1
echo "== tracer + planner, every arch";  $PY tools/arch_matrix.py build/archs --out build/archs/matrix.md | tail -1
[ "$1" = "--cpu-only" ] && exit 0
[ -f build/slots/qwen3moe-64x.gguf ] || $PY tools/make_test_model.py build/archs/qwen3moe-moe.gguf build/slots/qwen3moe-64x.gguf
echo "== planner slots = llama.cpp's";   $PY tools/check_slots.py build/slots/qwen3moe-64x.gguf | tail -1
echo "== cache + profile = CPU, every arch"; $PY tools/gpu_matrix.py build/archs --out build/archs/gpu_matrix.md | tail -1
echo "== the same, Q8_0 experts (the hybrid's one-token GPU path)"
$PY tools/gpu_matrix.py build/archs --quant-experts --out build/archs/gpu_matrix_q8.md | tail -1
G=~/models/granite/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf
echo "== audit records (--moe-cache-audit)"
if [ -f $G ] && [ -f build/granite/granite.strp ]; then $PY tools/check_audit.py $G build/granite/granite.strp; else echo "skipped: needs $G and build/granite/granite.strp"; fi
