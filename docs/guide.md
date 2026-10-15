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
- [Connect your tools](connect.md) (OpenAI, Claude Code)
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

    WSL also shuts its VM down a minute after the last program in it exits, which empties the file cache: the next
    start reads the whole model from disk again (about 80 s for gpt-oss-120b at 1 GB/s). A running `serve.py` keeps it
    alive. To keep the model cached between runs (or while benchmarking), leave something running, e.g.
    `wsl -d Ubuntu -- sleep infinity` in a spare terminal.
- **Disk:** the model, plus nothing big; profiles and traces are kilobytes to megabytes.
- **Software:** git, CMake 3.14+, a C++17 compiler, the CUDA toolkit (`nvcc`), Python 3.10+ with `venv` (on
  Ubuntu: `sudo apt install python3-venv`). Ninja is used if present.

## Install

```bash
git clone https://github.com/<you>/extended-strata-moe && cd extended-strata-moe
bash scripts/setup.sh
```

`scripts/setup.sh`:

1. clones llama.cpp into `./llama.cpp` (or uses `$LLAMA_DIR`), checks out the pinned commit in `patches/BASE` on a
   branch `extended-strata-moe`, and applies `patches/llama.cpp-extended-strata-moe.patch`;
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
git checkout -b extended-strata-moe (Get-Content ..\extended-strata-moe\patches\BASE)
git apply ..\extended-strata-moe\patches\llama.cpp-extended-strata-moe.patch
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

The binaries land in `build\bin\Release`. For the Python tools:

```powershell
cd ..\extended-strata-moe
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
cmake -S tools\routing_trace -B build -DLLAMA_DIR=..\llama.cpp
cmake --build build --config Release
$env:LLAMA_BIN = "..\llama.cpp\build\bin\Release"
```

The tools are tested on Linux and WSL.

## 1. Trace your text

**This step is optional.** The cache is adaptive: without a profile it starts empty and fills itself with the experts
your conversation uses (`--moe-cache-adaptive`, which `serve.py` turns on; [how it works](reference.md#the-adaptive-cache)).
On gpt-oss-120b that is 26.9 tokens/s with no profile. A profile starts the cache warm, and one made from text like yours
is the fastest setup (28.2); one made from other text is worse than none (24.8), so skip this step unless you have text
like yours.

The profile ranks experts by how often **your kind of text** routes to them, so trace text like what you will run:
your chats, your code, documents in your language. A few thousand tokens is enough; more gives a steadier ranking.
The match matters: with a fixed cache (no `--moe-cache-adaptive`) on gpt-oss-120b, a profile from llama.cpp's docs served
21% of the lookups from VRAM for a prompt of ordinary prose, and a profile from the same kind of prose 35%, which is the
difference between 22.5 and 26.0 tokens/s.

**In one command** (the tracer, then the planner; the profile does not depend on the cache size):

```bash
.venv/bin/python tools/make_profile.py MODEL.gguf my_text.txt -o model.strp
```

Then go to [3. Run](#3-run). The two steps by hand, for the planner's report:

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
.venv/bin/python tools/serve.py MODEL.gguf --profile model.strp --cache-mib N   # --profile and --cache-mib are optional
```

`tools/serve.py` sizes the cache to your GPU, starts `llama-server`, and prints the settings your tools need (an
OpenAI-compatible and an Anthropic-compatible endpoint on one port: [connect.md](connect.md)). You do not need the
planner's `--cache-mib` for it. It loads the model once without the cache to read how much VRAM the weights, the KV cache
and the compute buffer take (llama.cpp's own log), and gives the expert cache what is left:

```
GPU: 7910 of 8188 MiB free
  batch  4096: weights 1608 + KV 1305 + compute 3356 MiB, leaves 1241 MiB for the expert cache
  batch  2048: weights 1608 + KV 1233 + compute 1670 MiB, leaves 2999 MiB for the expert cache
```

The batch size (`--ubatch`, default `auto`) is the other half of that sum. A prompt is processed in batches, and every
batch uploads the experts of the whole model over PCIe (57 GiB for gpt-oss-120b, about 5.5 s), so a bigger batch means
a faster prompt: 95 tokens/s at 512, about 400 at 4096. Its compute buffer takes VRAM from the expert cache, which
speeds up generation. `auto` takes the biggest batch that leaves `--min-cache-mib` (1500) for the cache; set `--ubatch N`
to choose. Never give the cache more than is left: under WSL2 the driver does not fail an allocation that does not fit
(buffers adding up to 10 GB loaded without an error on an 8 GB card, with 7.9 GB shown as used); the excess is spilled
to system memory, which the GPU reads at PCIe speed.

The same thing by hand:

```bash
llama.cpp/build/bin/llama-server -m MODEL.gguf -ngl 999 -fit off --cpu-moe -c 32768 -b 4096 -ub 4096 -np 1 \
    --moe-cache-mib N --moe-cache-profile model.strp --moe-cache-hybrid --moe-cache-adaptive -t 10
```

- `--cpu-moe` keeps every expert in RAM; the cache then holds the profile's set in VRAM.
- `--moe-cache-adaptive` lets it change: while generating, it swaps the experts your text uses most into VRAM. Leave the
  profile out and it starts empty. Without the flag the cache keeps the profile's set for good.
- `-fit off` stops llama.cpp from re-fitting the memory around a cache it does not know about.
- **Threads matter, and more is not better.** On an i7-13620H (6 fast cores with 2 threads each, 4 slow cores; WSL
  reports 8 cores of 2 threads, which is wrong), gpt-oss-120b (a fixed cache, an earlier session: compare inside a sweep,
  not with the tables above) generates 26.4 tokens/s at `-t 6`, 29.2 at `-t 10`, 25.0 at `-t 12` and 15.6 at `-t 16`:
  past the fast cores, every step waits for the slowest thread. `serve.py` defaults to 5/8 of the logical CPUs; sweep
  your machine with `bench_cache.py -t` ([4. Measure](#4-measure)).
- At load, llama.cpp logs how many experts the profile filled, how many layers are held whole, and (CUDA, one token at
  a time) that the layers the cache holds in part hand over to the CPU [through a mailbox](reference.md#the-mailbox).

## 4. Measure

The point of the patch is speed, so measure it on your machine against the best layer split:

```bash
.venv/bin/python tools/bench_cache.py MODEL.gguf my_text.txt --profile model.strp \
    --cache-mib N --n-cpu-moe K --only-hybrid --adaptive -t 4 6 8 --runs 3 -n 400
```

- It runs every config at every thread count, in **interleaved rounds** (round 1 of all, then round 2, ...): a
  laptop's speed drifts with its temperature by more than the configs differ, so back-to-back runs mislead.
- `--adaptive` adds the adaptive cache, from the profile and from an empty cache. It learns as it generates, so give
  it `-n 400` or more: at the default 128 tokens it is judged on its first replies.
- Take `K` from the planner's "layer split in the same VRAM" line.
- Keep whichever config is faster at its own best thread count.

## Watch and audit

- `--metrics` on llama-server, then `tools/monitor.html` for a live dashboard (README, "Watch it live").
- `--moe-cache-audit run.jsonl` records the setup and every request; attach it to bug reports and benchmark
  issues. `tools/check_audit.py` validates one ([reference.md](reference.md#audit-log)).

## Troubleshooting

| symptom | cause, fix |
|---|---|
| `--moe-cache-profile, --moe-cache-hybrid, --moe-cache-adaptive and --moe-cache-audit need --moe-cache-mib N` | no cache size: add `--moe-cache-mib` |
| `--moe-cache-adaptive needs --moe-cache-hybrid` | add `--moe-cache-hybrid`: the upload cache (no hybrid) is an LRU, which follows the text by itself |
| `... made for 36 layers x 128 experts, the model has ...` | the profile is for another model: plan again |
| `MoE cache budget is too small for N layers, they are not cached` | the budget does not reach one slot per group: raise `--moe-cache-mib` |
| `hybrid MoE cache without a profile holds no experts` | add `--moe-cache-adaptive` (it fills itself) or `--moe-cache-profile` |
| very slow, disk busy | the model does not fit in RAM (WSL: see [What you need](#what-you-need)) |
| low hit rate in the server log | a fixed profile traced on different text (24% against 44% on gpt-oss-120b): add `--moe-cache-adaptive` so the cache follows your text, or `tools/make_profile.py` with yours |
| slower than `--n-cpu-moe` | compare at several `-t` before deciding (the layer split also gains from more threads); check that the start-up log (llama-server's, or `-v`) says the layers "hand over to the CPU through a mailbox" (CUDA, one token at a time) |
| the first start takes a minute or more, disk busy | the model is read from disk again: WSL emptied its file cache when it went idle (see [What you need](#what-you-need)) |
| the prompt is slow (100 tokens/s) | the batch size is 512: `serve.py` picks a bigger one, `--ubatch 2048` or `4096` by hand |
| generation slows down when I raise `--ubatch` | the batch's compute buffer took the expert cache's VRAM: `serve.py` shows the trade; use `--ubatch 2048` |
| `MoE cache: the GPU did not hand layer N over within 20 s` | the mailbox's flag never came. Set `LLAMA_MOE_CACHE_NO_MAILBOX=1` to use the scheduler's copies instead, and report it with the audit log |
| out of memory at load | lower `--moe-cache-mib`, or `-c` (the KV cache) |
| the machine stutters, the speed is erratic | the GPU's VRAM is over-committed and the driver is spilling it: lower `--moe-cache-mib` or `--ubatch` |
