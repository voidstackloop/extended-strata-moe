#!/usr/bin/env bash
# scripts/setup.sh - llama.cpp with the extended-strata-moe patch, the Python tools and the routing tracer (Linux, WSL, macOS).
#
#   bash scripts/setup.sh                  # CUDA if nvcc is on the PATH, else CPU only
#
# Environment: LLAMA_DIR (default: ./llama.cpp, cloned if missing), GGML_CUDA=ON|OFF, CUDA_ARCH (e.g. 89),
# JOBS (default: all cores), CMAKE_ARGS (more llama.cpp CMake options, e.g. -DGGML_HIP=ON).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
LLAMA_DIR=${LLAMA_DIR:-$REPO/llama.cpp}
BASE=$(cat "$REPO/patches/BASE")
PATCH=$REPO/patches/llama.cpp-extended-strata-moe.patch
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN)}
GEN=()
command -v ninja > /dev/null && GEN=(-G Ninja)

echo "== llama.cpp at ${BASE:0:7} + the extended-strata-moe patch: $LLAMA_DIR"
if [ ! -d "$LLAMA_DIR/.git" ]; then
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
if git apply --reverse --check "$PATCH" 2> /dev/null; then
    echo "the patch is applied already"
else
    # stops here if the checkout has changes of its own: commit or stash them first
    git checkout -B extended-strata-moe "$BASE"
    git apply "$PATCH"
fi

CUDA=${GGML_CUDA:-$(command -v nvcc > /dev/null && echo ON || echo OFF)}
echo "== build (CUDA $CUDA)"
# shellcheck disable=SC2086
cmake -S . -B build "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA="$CUDA" -DLLAMA_BUILD_TESTS=ON \
    ${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH} ${CMAKE_ARGS:-}
cmake --build build -j "$JOBS" --target llama-server llama-completion llama-cli llama-perplexity test-llama-archs

echo "== Python tools (.venv) and the routing tracer (build/)"
cd "$REPO"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
cmake -S tools/routing_trace -B build "${GEN[@]}" -DLLAMA_DIR="$LLAMA_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$JOBS"

echo
echo "done:"
echo "  $LLAMA_DIR/build/bin/llama-server   (and llama-completion, llama-cli)"
echo "  $REPO/build/routing_trace"
echo "next: the README's 'Quick start' - serve.py (make_profile.py for your kind of text is optional)"
