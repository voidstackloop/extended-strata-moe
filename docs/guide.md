# Guide

From a MoE GGUF to a measured, tuned setup. The [README](../README.md) has the short version;
[reference.md](reference.md) has every flag and format.

- [What you need](#what-you-need)
- [Install](#install)
- [Windows](#windows)
- [1. Trace your text](#1-trace-your-text)
- [2. Plan the cache](#2-plan-the-cache)
- [3. Run](#3-run)
- [4. Measure](#4-measure)
- [Watch and audit](#watch-and-audit)
- [Troubleshooting](#troubleshooting)

## What you need

- **An NVIDIA GPU** (CUDA). AMD (ROCm) should work but is untested. The hybrid also runs on other backends, but
  slower (see the FAQ in the README).
- **RAM for the whole model.** The experts are read from RAM every token. If the model does not fit in RAM, the OS
  pages experts in from disk and the speed collapses: gpt-oss-120b (59 GiB) wants 64 GB or more.
  - **WSL2 users:** WSL takes half of the PC's RAM by default. Raise it in `%UserProfile%\.wslconfig`:

    ```ini
    [wsl2]
    memory=80GB
    swap=4GB
    ```

    then run `wsl --shutdown` from Windows and start WSL again.
- **Disk:** the model, plus nothing big; profiles and traces are kilobytes to megabytes.
- **Software:** git, CMake 3.14+, a C++17 compiler, the CUDA toolkit (`nvcc`), Python 3.10+ with `venv` (on
  Ubuntu: `sudo apt install python3-venv`). Ninja is used if present.

## Install

```bash
git clone https://github.com/<you>/strata-moe && cd strata-moe
bash scripts/setup.sh
```

`scripts/setup.sh`:

1. clones llama.cpp into `./llama.cpp` (or uses `$LLAMA_DIR`), checks out the pinned commit in `patches/BASE` on a
   branch `strata-moe`, and applies `patches/llama.cpp-strata-moe.patch`;
2. builds llama-server, llama-completion, llama-cli, llama-perplexity and test-llama-archs, with CUDA when `nvcc`
   is on the PATH;
3. creates `.venv` with the Python tools' two dependencies (`gguf`, `numpy`);
4. builds the routing tracer into `build/` (a static, CPU-only llama.cpp).

Options, as environment variables:

| variable | default | |
|---|---|---|
| `LLAMA_DIR` | `./llama.cpp` | an existing llama.cpp checkout to use; it must have no local changes |
| `GGML_CUDA` | `ON` if `nvcc` is found | `OFF` for a CPU-only build |
| `CUDA_ARCH` | CMake's choice | e.g. `89` (RTX 40), `86` (RTX 30), `120` (RTX 50): a faster build |
| `CMAKE_ARGS` | | more llama.cpp options, e.g. `-DGGML_HIP=ON` |
| `JOBS` | all cores | |

Running it again is safe: it skips the clone and the patch when they are done, and rebuilds.

## Windows

The patch is ordinary llama.cpp code, so it builds like llama.cpp does on Windows: Visual Studio 2022 with the C++
workload, CMake, and the CUDA toolkit for Windows. **A native Windows build has not been tested yet**; please open
an issue with what you find.

```powershell
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout -b strata-moe (Get-Content ..\strata-moe\patches\BASE)
git apply ..\strata-moe\patches\llama.cpp-strata-moe.patch
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

The binaries land in `build\bin\Release`. For the Python tools:

```powershell
cd ..\strata-moe
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
cmake -S tools\routing_trace -B build -DLLAMA_DIR=..\llama.cpp
cmake --build build --config Release
$env:LLAMA_BIN = "..\llama.cpp\build\bin\Release"
```

The tools are tested on Linux and WSL.

## 1. Trace your text

The profile ranks experts by how often **your kind of text** routes to them, so trace text like what you will run:
your chats, your code, documents in your language. A few thousand tokens is enough; more gives a steadier ranking.

```bash
build/routing_trace MODEL.gguf my_text.txt trace.bin 4096 8     # n_ctx, CPU threads
```

- It runs on the CPU and needs no GPU. It prints how many tokens and routing records it wrote.
- For a held-out check of the hit rate (step 2), trace a second, different text the same way.
- Split GGUFs: pass the first shard (`*-00001-of-0000N.gguf`).

## 2. Plan the cache

```bash
.venv/bin/python tools/moe_plan.py MODEL.gguf --vram-gib 8 --trace trace.bin --out model.strp
```

- `--vram-gib G`: your GPU's memory. The cache gets what the dense weights (attention, embeddings, output) and
  `--reserve-gib` (default 2: KV cache, compute buffers, the driver) leave. `--cache-mib N` sets the size directly.
- The output lists each group of layers with its slots, the predicted hit rate on your trace, the llama.cpp flags
  to use, and the `--n-cpu-moe` layer split that fits the same VRAM: the baseline to beat.
- `--whole-layers N` ranks every expert of the last N layers first. Layers held whole run on the GPU with no CPU
  round trip; the rest of the budget still goes to single experts. Worth trying when the experts are small.

To see how well the profile generalizes, replay held-out text through every policy at llama.cpp's exact slot counts:

```bash
.venv/bin/python tools/sim_cache.py MODEL.gguf --cache-mib 3000 4800 --train trace.bin --test heldout.bin
```

The "static" column is the hybrid's hit rate; "layer split" is `--n-cpu-moe`'s.

## 3. Run

```bash
llama.cpp/build/bin/llama-server -m MODEL.gguf -ngl 999 -fit off --cpu-moe \
    --moe-cache-mib N --moe-cache-profile model.strp --moe-cache-hybrid -t 6
```

- `--cpu-moe` keeps every expert in RAM; the cache then holds the profile's set in VRAM.
- `-fit off` stops llama.cpp from re-fitting the memory around a cache it does not know about.
- **Threads matter.** The hybrid and the layer split have different best thread counts. On an i7-13620H (6
  performance cores) both were best at `-t 6`. Start at your performance-core count and try ±2.
- If loading fails out of memory, lower `--moe-cache-mib` (or plan again with a larger `--reserve-gib`).
- At load, llama.cpp logs how many experts the profile filled and how many layers are held whole.

## 4. Measure

The point of the patch is speed, so measure it on your machine against the best layer split:

```bash
.venv/bin/python tools/bench_cache.py MODEL.gguf my_text.txt --profile model.strp \
    --cache-mib N --n-cpu-moe K --hybrid --only-hybrid -t 4 6 8 --runs 3
```

- It runs every config at every thread count, in **interleaved rounds** (round 1 of all, then round 2, ...): a
  laptop's speed drifts with its temperature by more than the configs differ, so back-to-back runs mislead.
- Take `K` from the planner's "layer split in the same VRAM" line.
- Keep whichever config is faster at its own best thread count.

## Watch and audit

- `--metrics` on llama-server, then `tools/monitor.html` for a live dashboard (README, "Watch it live").
- `--moe-cache-audit run.jsonl` records the setup and every request; attach it to bug reports and benchmark
  issues. `tools/check_audit.py` validates one ([reference.md](reference.md#audit-log)).

## Troubleshooting

| symptom | cause, fix |
|---|---|
| `--moe-cache-profile, --moe-cache-hybrid and --moe-cache-audit need --moe-cache-mib N` | no cache size: add `--moe-cache-mib` |
| `... made for 36 layers x 128 experts, the model has ...` | the profile is for another model: plan again |
| `MoE cache budget is too small for N layers, they are not cached` | the budget does not reach one slot per group: raise `--moe-cache-mib` |
| `hybrid MoE cache without a profile holds no experts` | add `--moe-cache-profile` |
| very slow, disk busy | the model does not fit in RAM (WSL: see [What you need](#what-you-need)) |
| low hit rate in the server log | the profile was traced on different text: trace yours |
| slower than `--n-cpu-moe` | expected for small experts and fast round trips; compare at several `-t` before deciding |
| out of memory at load | lower `--moe-cache-mib`, or `-c` (the KV cache) |
