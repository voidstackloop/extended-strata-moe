# Changelog

## Unreleased: Tessera-125B-MoE "k448" support files

No engine (C++) change. Added `models/k448/` (Windows start scripts for a 128K profile and a 64K + MTP profile, a patched chat template so Claude Code
works without a proxy, a CPU expert-kernel microbenchmark and decode/prompt speed tools) and `docs/k448.md` with the measurements on an RTX 4060 Laptop:
17.5 tok/s with a small expert cache, 20 to 22.5 with a 3.2 GB cache, 20 to 21.5 with MTP at 64K context.

## 0.1.0 - unreleased

First public version: [Strata](https://github.com/Niko1221/Strata)'s expert tiering for every llama.cpp MoE model.
llama.cpp `de7fa0a` + `patches/llama.cpp-extended-strata-moe.patch`.

**llama.cpp**

- `--moe-cache-profile FILE`: fill the MoE cache at load from a Strata `STRP` profile, hottest experts first.
- `--moe-cache-hybrid`: Strata's hybrid. A profile-ranked set of experts in VRAM; the CPU computes the
  others from RAM in the same step; nothing is uploaded during generation. Layers held whole skip the CPU.
- `--moe-cache-adaptive`: the hybrid's cache follows the conversation. It counts the experts the router picks and, every
  32 generated tokens, swaps the hottest ones the cache lacks into the slots of the coldest it holds (at most 64, only when
  clearly hotter). It needs no profile (it starts empty and fills itself) and improves on a good one. gpt-oss-120b on an
  8 GB card, 400 tokens from a cold start: 26.9 tokens/s with no profile (the best layer split: 15.9, a fixed cache with a
  profile from like text: 26.0), 28.2 with a profile from like text. A profile from other text is worse than none there.
  `llama_moe_cache_rebalance`, `n_swap`, the `llamacpp:moe_cache_swaps_total` series and `swaps` in the audit log.
  `test-llama-archs` has an adaptive config: tokens one at a time with swaps between chunks, then a prompt batch against
  the CPU's logits; it fails if swapping is broken (seen to fail on three deliberate breaks).
- The CPU's `mul_mat_id` and CUDA's one-token `mul_mat_vec_q` skip rows with expert id -1.
- **The mailbox** (CUDA, one token at a time): the GPU and the CPU hand each layer over through page-locked host memory
  (`ggml/include/ggml-mailbox.h`) and work on it at the same time, in place of the scheduler's synchronous copies
  (130 to 180 µs a layer under WSL2). A hand-over measures 3 µs. A new scheduler flag, `GGML_TENSOR_FLAG_NO_SYNC`, lets the
  CPU's split start while the GPU's is still running. Generation on gpt-oss-120b: 17.2 to 21.3 tokens/s.
- `GGML_SCHED_PROFILE=1`: where a generated token's time goes (CPU compute, GPU, waits and copies).
- `--moe-cache-audit FILE`: a JSONL audit trail in every tool that loads through `common`: load (model, flags,
  GPUs, load time, the profile's SHA-256, the experts held per layer), exit (counters and the run's throughput);
  llama-server and llama-cli also ready and one record per request. Records carry a millisecond UTC time and the pid.
- `llama_moe_cache_get_stats`, `llama_moe_cache_get_layer_stats`, `llama_moe_cache_get_resident`.
- llama-server: `llamacpp:moe_cache_*` Prometheus series (totals and per layer), and a per-request log line.
- `test-llama-archs`: `--all-host-experts` and `--quant-experts`; cache, profile and hybrid configs with
  bit-identity and first-MoE-layer checks.

**Tools**

- `routing_trace`, `moe_plan.py`, `sim_cache.py`, `bench_cache.py` (interleaved, thread sweeps), `monitor.html`,
  `check_audit.py`, and the check suite (`check_all.sh`).
- `scripts/setup.sh`: clone, patch and build in one step.
- `serve.py` starts llama-server with the adaptive hybrid, **sized to the GPU by measurement** (the weights, KV cache and
  compute buffer of the batch size, then the cache gets what is left) and prints the settings for OpenAI-compatible tools
  and for Claude Code (`docs/connect.md`); `check_api.py` tests both APIs, auth and the cache. `--profile` is optional.
- `make_profile.py`: the profile for your text in one command.
- `serve.py --favor generation`: the biggest batch whose expert cache is within 3% of the largest (faster answers, slower prompts).
- The prefill and the adaptive swaps copy experts through a pinned staging buffer, queued on the device's stream:
  0.47 -> 0.29 ms a swap on Qwen3-Coder-Next.
- At exit (`-v`), how long the CPU waited on average for each mailbox hand-over: the GPU's part of a layer.
- The CPU's one-token expert products (`mul_mat_id`) use small chunks of 32 rows, taken by the threads as they come,
  instead of one slice per thread: a hybrid CPU's slow cores no longer hold every expert up. Qwen3-Coder-Next Q6_K on an
  i7-13620H: 115-120 -> 92-96 us an op, 36.0 -> 33.7 ms a token over 300 tokens; 26.85 ms (37.2 tokens/s) over 1,200.
  `GGML_MMID_CHUNK` sets the rows, 0 the old split. `GGML_CPU_OP_PROFILE=1` prints the CPU's time per op kind.
- `tools/code_eval.py` (six coding tasks through `serve.py`, answers run against asserts; `--suite hard` adds ten harder ones), and `route_dump` +
  `tools/route_predict.py` (how well a layer's experts can be guessed before its attention).
- Prompt speed: a bigger batch uploads the experts once per batch. gpt-oss-120b: 95 tokens/s at the default batch
  size of 512, about 400 at 4096; `serve.py --ubatch auto` picks the biggest that leaves room for the cache.

**Fixed before release**

- The hybrid left the experts' biases in host memory (gpt-oss has them), so each bias sent the graph back to the CPU:
  4 GPU-CPU round trips a layer instead of 1. The cache now holds device copies of the biases; the graph splits go
  from 194 to 50 on gpt-oss-20b, the same as `--cpu-moe`.
