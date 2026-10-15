# Reference

Everything extended-strata-moe adds to llama.cpp, and the formats its tools read and write. The code is in
`patches/llama.cpp-extended-strata-moe.patch` (against llama.cpp `patches/BASE`) and `tools/`.

- [Command-line options](#command-line-options)
- [C API](#c-api)
- [Server metrics](#server-metrics)
- [Log lines](#log-lines)
- [Audit log](#audit-log)
- [Profile format (STRP)](#profile-format-strp)
- [Routing trace format](#routing-trace-format)
- [Tools](#tools)

## Command-line options

For every llama.cpp tool built on `common` (llama-server, llama-cli, llama-completion, llama-perplexity, ...).

| option | env | meaning |
|---|---|---|
| `--moe-cache-mib N` | `LLAMA_ARG_MOE_CACHE_MIB` | upstream: an N MiB VRAM cache for the experts kept in RAM. Needed by all the options below |
| `--moe-cache-profile FILE` | `LLAMA_ARG_MOE_CACHE_PROFILE` | fill the cache at load from a [profile](#profile-format-strp), hottest experts first. A profile made for another `n_layer x n_expert` is refused with a warning, and the cache starts empty |
| `--moe-cache-hybrid` | `LLAMA_ARG_MOE_CACHE_HYBRID` | with a profile: the cached experts stay (unless `--moe-cache-adaptive` swaps them), and the CPU computes the experts the cache does not hold. Nothing is uploaded during generation. Ubatches of up to 32 tokens take this path; larger ones (prompts) keep llama.cpp's |
| `--moe-cache-adaptive` | `LLAMA_ARG_MOE_CACHE_ADAPTIVE` | with `--moe-cache-hybrid`: [follow the conversation](#the-adaptive-cache). The profile only decides the start and may be left out: the cache then starts empty and fills itself |
| `--moe-cache-audit FILE` | `LLAMA_ARG_MOE_CACHE_AUDIT` | append the [audit log](#audit-log) to FILE |

The cache only holds experts that are in RAM, so it is used with `--cpu-moe` or `--n-cpu-moe N` (or `-ot` patterns
that keep `ffn_*_exps` on the CPU). Use `-ngl 999 -fit off`: without `-fit off`, llama.cpp re-fits unset parameters
to free memory, around a cache it does not know about.

Without `--moe-cache-mib`, the new options do nothing, and llama.cpp warns about it. `--moe-cache-adaptive` without
`--moe-cache-hybrid` is ignored with a warning: the upload cache is an LRU, which already follows the text.

### Environment variables

| variable | meaning |
|---|---|
| `LLAMA_MOE_CACHE_NO_MAILBOX=1` | one-token generation with the hybrid normally hands each layer over between the GPU and the CPU through a [mailbox](#the-mailbox) (CUDA). This turns it off and uses the scheduler's synchronous copies, as before. For comparing, or if a driver misbehaves |
| `GGML_MMID_CHUNK=N` | one-token expert products on the CPU are cut into chunks of N rows (default 32), taken by the threads as they come, so the fast cores do more of a hybrid CPU's work; 0 restores llama.cpp's one slice per thread |
| `GGML_CPU_OP_PROFILE=1` | at exit, the CPU backend's time per op kind (thread 0, the op and the barrier after it) |
| `GGML_SCHED_PROFILE=1` | on exit, prints where a generated token's time went: the CPU's compute, the GPU's, and the waits and copies between them, by graph shape (printed to stderr) |

### The mailbox

Under CUDA, in the hybrid's one-token decode, each layer the cache does not hold whole is a hand-over between the GPU
and the CPU. Through llama.cpp's scheduler that is two synchronous copies out, two or three back, and a stream
synchronization: 130 to 180 µs a layer under WSL2 (measured on gpt-oss-120b with `GGML_SCHED_PROFILE`). The mailbox
replaces it with page-locked host memory the GPU can write and the CPU poll:

1. The GPU looks up which of the token's experts the cache holds (it keeps its own copy of the slot map), stores the layer's
   input and the selected experts in the mailbox, and raises a flag stamped with the step's number.
2. The GPU goes on with the experts the cache holds, while the CPU, already running, sees the flag (a few µs), computes
   the others and stores the result in the mailbox.
3. The GPU reads the result and sums.

The mailbox is used for one token at a time, quantized experts and CUDA (NVIDIA; not ROCm), for the layers the cache holds
only in part. Everything else keeps the scheduler's path. It is described in `ggml/include/ggml-mailbox.h`.

### The adaptive cache

With `--moe-cache-adaptive` the set of experts in VRAM is not fixed at load. The hybrid counts the experts the router
picks (a heat per layer and expert, once per expert and ubatch; old picks fade, with a half-life of about 350 generated
tokens). Every 32 generated tokens, between two tokens while the GPU is idle, it swaps the hottest experts the cache
lacks in for the coldest ones it holds, at most 64 of them. A swap is a blocking copy from RAM into the evicted expert's
slot (gpt-oss-120b: 13 MB, 1.2 ms). An expert only comes in when its heat is more than 1.5 times the one it replaces
(plus two picks), so the set does not flap. Slots that are still free (a start without a profile, or a profile smaller
than the cache) are filled first: every 8 tokens, up to 128 at a time. Layers the cache holds whole are left alone.
Only ubatches of up to 32 tokens count, those that take the hybrid's path: a big prompt batch does not. The swaps happen
before a token starts, so they never touch what it computes.

What it is for: it needs no profile (the cache fills itself in a few hundred tokens), a profile made from other text is
corrected as the conversation goes on, and one made from the right text is improved on. The numbers are in the README's
"Results". It costs the copies: a few hundred experts while the cache fills (under a second on gpt-oss-120b), a handful
per round after.
The counters: `swaps` in the audit log, `llamacpp:moe_cache_swaps_total`, and a line at exit,
`llama_moe_cache: adaptive: N experts swapped in`.

`llama_moe_cache_rebalance` (see the [C API](#c-api)) does a round on request, for tools that decode in chunks.

## C API

`include/llama.h`. All of it is marked `[EXPERIMENTAL]`, like upstream's MoE cache.

```c
struct llama_context_params {
    // ...
    size_t       moe_cache_size;    // upstream: bytes per device, 0 = no cache
    const char * moe_cache_profile; // a profile to fill the cache from, nullptr = start empty
    bool         moe_cache_hybrid;  // the CPU computes the experts the cache does not hold
    bool         moe_cache_adaptive; // with the hybrid: swap the experts the router picks most into the cache while generating
};

struct llama_moe_cache_stats {
    uint64_t n_hit;          // selected experts the cache held (ubatches of up to 32 tokens)
    uint64_t n_hit_profile;  // of them, experts the profile put there and that were never evicted since
    uint64_t n_miss;         // selected experts the cache did not hold: uploaded, or computed on the CPU (hybrid)
    uint64_t n_bytes_upload; // bytes of those uploads
    uint64_t n_bytes_cpu;    // hybrid: bytes of the experts the CPU computed
    bool     hybrid;
    uint64_t n_copy;         // experts large batches copied from the cache instead of from RAM
    uint64_t n_bytes_copy;
    uint64_t n_slots;        // expert slots, all devices
    uint64_t n_prefilled;    // slots the profile filled at load
    uint64_t n_bytes_cache;  // device memory of the cache
    uint64_t n_swap;         // adaptive: experts swapped into the cache
};

// false (stats untouched) when the context has no MoE cache
bool    llama_moe_cache_get_stats(const struct llama_context * ctx, struct llama_moe_cache_stats * stats);

// one entry per layer the cache serves: {il, n_slots, n_hit, n_hit_profile, n_miss}
// fills at most n_max and returns how many there are; n_max = 0 sizes the array
int32_t llama_moe_cache_get_layer_stats(const struct llama_context * ctx, struct llama_moe_cache_layer_stats * stats, int32_t n_max);

// the expert ids the cache holds in layer il now; same sizing convention
int32_t llama_moe_cache_get_resident(const struct llama_context * ctx, int32_t il, int32_t * experts, int32_t n_max);

// adaptive cache: swap in at most max_swaps of the hottest experts the cache lacks for the coldest it holds, now
// (llama_decode does it by itself every few dozen tokens). Waits for the device, so call it between decodes.
// Returns the number of swaps, or -1 when the context has no adaptive MoE cache.
int32_t llama_moe_cache_rebalance(struct llama_context * ctx, int32_t max_swaps);
```

The counters run from context creation. "Hits" and "misses" count distinct experts per ubatch and layer.

## Server metrics

`llama-server --metrics` adds these to `/metrics` when the context has a MoE cache:

| series | type | meaning |
|---|---|---|
| `llamacpp:moe_cache_hits_total` | counter | `n_hit` |
| `llamacpp:moe_cache_hits_profile_total` | counter | `n_hit_profile` |
| `llamacpp:moe_cache_misses_total` | counter | `n_miss` |
| `llamacpp:moe_cache_upload_bytes_total` | counter | `n_bytes_upload` |
| `llamacpp:moe_cache_cpu_bytes_total` | counter | `n_bytes_cpu` |
| `llamacpp:moe_cache_copies_total` | counter | `n_copy` |
| `llamacpp:moe_cache_swaps_total` | counter | `n_swap` (adaptive) |
| `llamacpp:moe_cache_hit_ratio` | gauge | hits / (hits + misses) since the start |
| `llamacpp:moe_cache_slots` | gauge | `n_slots` |
| `llamacpp:moe_cache_prefilled` | gauge | `n_prefilled` |
| `llamacpp:moe_cache_bytes` | gauge | `n_bytes_cache` |
| `llamacpp:moe_cache_hybrid` | gauge | 1 with `--moe-cache-hybrid` |
| `llamacpp:moe_cache_layer_hits_total{layer="N"}` | counter | per layer |
| `llamacpp:moe_cache_layer_hits_profile_total{layer="N"}` | counter | per layer |
| `llamacpp:moe_cache_layer_misses_total{layer="N"}` | counter | per layer |

`tools/monitor.html` reads them (see the README's "Watch it live").

## Log lines

At load:

```
prefill: MoE cache filled with N experts (X MiB) from model.strp
impl: hybrid MoE cache: N layers held whole (GPU only), the others split between GPU and CPU
impl: adaptive hybrid MoE cache without a profile starts empty and fills itself while generating     # --moe-cache-adaptive
```

llama-server, per request:

```
MoE cache hits = R % (hybrid: H on the GPU, M on the CPU, which read X MiB)                  # hybrid
MoE cache hits = R % (H hit, P from the profile, M uploaded: X MiB, Y GiB/s)                 # upload mode
```

At exit (`-v`): totals for small (≤ 8 tokens) and large ubatches, and the spread of the per-layer hit rate; with
`--moe-cache-adaptive` also `llama_moe_cache: adaptive: N experts swapped in`. Each round of swaps is logged at debug
level (`-v`): `do_rebalance: 64 experts swapped in 94.5 ms`. With the mailbox:
`llama_moe_cache: mailbox: the CPU waited X us on average for each of N hand-overs from the GPU`, the time the GPU spends on
a layer before its experts (attention and router), which the CPU cannot overlap.

## Audit log

`--moe-cache-audit FILE` appends one JSON object per line (JSONL). Each record is written with a single
open-append-close, so the log survives a crash up to the last record, and several runs can share one file.

Every record has:

| field | type | |
|---|---|---|
| `event` | string | `load`, `ready`, `request` or `exit` |
| `time` | string | UTC, ISO 8601, to the millisecond |
| `pid` | int | the process (runs sharing one file tell apart by it) |
| `hybrid` | bool | |
| `hits`, `hits_profile`, `misses` | int | the counters since the context was created ([C API](#c-api)) |
| `bytes_upload`, `bytes_cpu`, `copies`, `bytes_copy`, `swaps` | int | the same (`swaps`: experts the adaptive cache swapped in) |

The events:

- **`load`**: written right after the context is created, before any warmup. Adds:
  - `model`: `{path, bytes, name, n_layer}`;
  - `params`: `{build, moe_cache_mib, moe_cache_hybrid, moe_cache_adaptive, n_gpu_layers, n_threads, n_threads_batch, n_ctx,
    n_batch, n_ubatch, tensor_overrides}` (`tensor_overrides` are the `--cpu-moe`/`--n-cpu-moe`/`-ot` patterns);
  - `cache`: `{slots, prefilled, bytes}`;
  - `load_ms`: the model's load time;
  - `devices`: per GPU, `{name, description, mib_free, mib_total}`, with the model loaded (empty on a CPU build);
  - `profile`: `{path, bytes, sha256}`, `{path, error}` if unreadable, or `null`;
  - `layers`: per cached layer, `{il, slots, hits, hits_profile, misses, resident}`, where `resident` is the list of
    expert ids the cache holds.
- **`ready`** (llama-server, and llama-cli, which runs the server in-process): once the model is loaded. The server's
  own decodes at load (the warmup, a context probe) fall between `load` and `ready`.
- **`request`** (llama-server, llama-cli): one per finished request. Adds `slot`, `task`, `n_prompt`, `n_gen`, `t_prompt_ms`,
  `t_gen_ms`, and the request's own counters: `req_hits`, `req_hits_profile`, `req_misses`, `req_bytes_upload`,
  `req_bytes_cpu`. With several slots busy at once, a request's counters include the others' lookups (the counters
  are the context's).
- **`exit`**: when the context is freed (normal exit, or the server unloading the model), in every tool that loads
  through `common` (llama-completion, llama-perplexity, llama-imatrix, llama-mtmd-cli, ...). Adds `layers` as in
  `load`, and `perf`: `{n_prompt, n_gen, t_prompt_ms, t_gen_ms, prompt_tps, gen_tps}`, the run's throughput after the
  warmup (llama.cpp's own `llama_perf_context`). A killed process writes no `exit` record.

With one slot, the `request` records add up exactly to `exit` minus `ready`. `tools/check_audit.py` checks this and
the rest:

```bash
.venv/bin/python tools/check_audit.py MODEL.gguf PROFILE.strp [--cache-mib 256]
```

## Profile format (STRP)

Strata's `write_expert_profile` layout, little-endian:

| offset | type | |
|---|---|---|
| 0 | 4 bytes | magic `STRP` |
| 4 | u32 | version, 1 |
| 8 | u32 | `n_layers` |
| 12 | u32 | `n_expert` |
| 16 | u32 | slots the profile was made for (informational; llama.cpp fills as many as it has) |
| 20 | u32 | `n_ranked` |
| 24 | `n_ranked` x (u16 layer, u16 expert) | the pairs, hottest first |
| ... | `n_layers x n_expert` x i32 | each pair's rank, -1 if unranked (llama.cpp does not read it) |

llama.cpp validates the magic, the version, the dimensions against the model, `n_ranked <= n_layers x n_expert`,
and every pair's range. Each group of layers (layers with the same expert tensor types and shapes) takes the
profile's pairs of its layers in order, as many as it has slots.

## Routing trace format

Strata's `--dump-routing` layout, written by `tools/routing_trace`: one record per token and MoE layer,

```
int32 layer, int32 k, k x int32 expert id, k x float32 weight
```

`routing_trace` writes the weights as 0; nothing downstream reads them.

## Tools

| tool | usage |
|---|---|
| `build/routing_trace` | `routing_trace MODEL.gguf TEXT.txt OUT.bin [n_ctx=4096] [n_threads=4]` |
| `tools/moe_plan.py` | `moe_plan.py MODEL.gguf [shards...] (--vram-gib G [--reserve-gib 2] \| --cache-mib N) [--trace T.bin ...] [--whole-layers N] [--out P.strp]` |
| `tools/sim_cache.py` | `sim_cache.py MODEL.gguf --cache-mib N [N...] --train T.bin --test H.bin [--pin-reserve F]` |
| `tools/bench_cache.py` | `bench_cache.py MODEL.gguf TEXT --profile P.strp [--cache-mib N...] [--n-cpu-moe N...] [--hybrid \| --only-hybrid] [-t T...] [-n 128] [--runs 2]` |
| `tools/make_profile.py` | `make_profile.py MODEL.gguf TEXT [-o P.strp] [--threads 8] [--ctx 4096]`: the tracer and the planner in one command |
| `tools/code_eval.py` | `code_eval.py MODEL.gguf [--port 8099] [--max-tokens 700] [--suite smoke\|hard] [--no-server]`: six (`smoke`) or ten harder (`hard`) Python tasks through `serve.py`, each answer run against asserts; a smoke test of a quantization's coding |
| `routing_trace/route_dump` + `tools/route_predict.py` | `route_dump MODEL TEXT OUT.bin [n_tokens] [threads]`, then `route_predict.py MODEL OUT.bin`: how well a layer's experts can be guessed before its attention |
| `tools/serve.py` | `serve.py MODEL.gguf [--profile P.strp] [--ubatch auto\|N] [--favor prompt\|generation] [--min-cache-mib 1500] [--cache-mib N] [--margin-mib 400] [--api-key K] [--alias NAME] [--host H] [--port 8080] [--ctx 32768] [--parallel 1] [--threads auto] [--dry-run] [-- llama-server flags]`: fits the cache and the batch size to the GPU ([connect.md](connect.md)) |
| `tools/check_api.py` | `check_api.py MODEL.gguf PROFILE.strp [--cache-mib 256]`: the OpenAI and Anthropic endpoints, auth, and the cache |
| `tools/check_audit.py` | `check_audit.py MODEL.gguf PROFILE.strp [--cache-mib 256]`: the audit records, and that the mailbox changes nothing in the text |
| `tools/monitor.html` | served over HTTP: `monitor.html?server=http://127.0.0.1:8080` |
| `tools/check_all.sh` | every check, CPU then GPU (`--cpu-only` to stop after the CPU ones) |

The tools find the llama.cpp binaries in `$LLAMA_BIN`, else `llama.cpp/build/bin` in this repo
(`scripts/setup.sh`), else `~/projects/llama.cpp/build-cuda/bin`.
