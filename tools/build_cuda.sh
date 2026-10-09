#!/bin/bash
# tools/build_cuda.sh - the extended-strata-moe branch of llama.cpp with CUDA, from the user-local toolkit (~/cuda-13.4, no root).
#   CUDA_ARCH=89 tools/build_cuda.sh [targets...]     (default: test-llama-archs llama-cli llama-server)
set -e
CUDA=${CUDA:-$HOME/cuda-13.4}
LLAMA=${LLAMA:-$HOME/projects/llama.cpp}
# without WSL's Windows PATH entries: CMake's program searches over /mnt/c (9P) stall the configure for minutes
export PATH="$CUDA/bin:$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd:)"
export LD_LIBRARY_PATH="$CUDA/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
cmake -S "$LLAMA" -B "$LLAMA/build-cuda" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-89}" -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF
targets=("$@")
[ ${#targets[@]} -gt 0 ] || targets=(test-llama-archs llama-cli llama-server)
cmake --build "$LLAMA/build-cuda" -j "${JOBS:-12}" --target "${targets[@]}"
