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
| `--moe-cache-hybrid` | `LLAMA_ARG_MOE_CACHE_HYBRID` | with a profile: the cached experts stay for good, and the CPU computes the experts the cache does not hold. Nothing is uploaded during generation. Ubatches of up to 32 tokens take this path; larger ones (prompts) keep llama.cpp's |
| `--moe-cache-audit FILE` | `LLAMA_ARG_MOE_CACHE_AUDIT` | append the [audit log](#audit-log) to FILE |

The cache only holds experts that are in RAM, so it is used with `--cpu-moe` or `--n-cpu-moe N` (or `-ot` patterns
that keep `ffn_*_exps` on the CPU). Use `-ngl 999 -fit off`: without `-fit off`, llama.cpp re-fits unset parameters
to free memory, around a cache it does not know about.

Without `--moe-cache-mib`, the three new options do nothing, and llama.cpp warns about it.

## C API

`include/llama.h`. All of it is marked `[EXPERIMENTAL]`, like upstream's MoE cache.

```c
struct llama_context_params {
    // ...
    size_t       moe_cache_size;    // upstream: bytes per device, 0 = no cache
    const char * moe_cache_profile; // a profile to fill the cache from, nullptr = start empty
    bool         moe_cache_hybrid;  // the CPU computes the experts the cache does not hold
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
};

// false (stats untouched) when the context has no MoE cache
bool    llama_moe_cache_get_stats(const struct llama_context * ctx, struct llama_moe_cache_stats * stats);

// one entry per layer the cache serves: {il, n_slots, n_hit, n_hit_profile, n_miss}
// fills at most n_max and returns how many there are; n_max = 0 sizes the array
int32_t llama_moe_cache_get_layer_stats(const struct llama_context * ctx, struct llama_moe_cache_layer_stats * stats, int32_t n_max);

// the expert ids the cache holds in layer il now; same sizing convention
int32_t llama_moe_cache_get_resident(const struct llama_context * ctx, int32_t il, int32_t * experts, int32_t n_max);
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
```

llama-server, per request:

```
MoE cache hits = R % (hybrid: H on the GPU, M on the CPU, which read X MiB)                  # hybrid
MoE cache hits = R % (H hit, P from the profile, M uploaded: X MiB, Y GiB/s)                 # upload mode
```

At exit (`-v`): totals for small (≤ 8 tokens) and large ubatches, and the spread of the per-layer hit rate.

## Audit log

`--moe-cache-audit FILE` appends one JSON object per line (JSONL). Each record is written with a single
open-append-close, so the log survives a crash up to the last record, and several runs can share one file.

Every record has:

| field | type | |
|---|---|---|
| `event` | string | `load`, `ready`, `request` or `exit` |
| `time` | string | UTC, ISO 8601 |
| `hybrid` | bool | |
| `hits`, `hits_profile`, `misses` | int | the counters since the context was created ([C API](#c-api)) |
| `bytes_upload`, `bytes_cpu`, `copies`, `bytes_copy` | int | the same |

The events:

- **`load`**: written right after the context is created, before any warmup. Adds:
  - `model`: `{path, bytes, name, n_layer}`;
  - `params`: `{build, moe_cache_mib, moe_cache_hybrid, n_gpu_layers, n_threads, n_threads_batch, n_ctx,
    n_batch, n_ubatch, tensor_overrides}` (`tensor_overrides` are the `--cpu-moe`/`--n-cpu-moe`/`-ot` patterns);
  - `cache`: `{slots, prefilled, bytes}`;
  - `profile`: `{path, bytes, sha256}`, `{path, error}` if unreadable, or `null`;
  - `layers`: per cached layer, `{il, slots, hits, hits_profile, misses, resident}`, where `resident` is the list of
    expert ids the cache holds.
- **`ready`** (llama-server): once the model is loaded. The server's own decodes at load (the warmup, a context
  probe) fall between `load` and `ready`.
- **`request`** (llama-server): one per finished request. Adds `slot`, `task`, `n_prompt`, `n_gen`, `t_prompt_ms`,
  `t_gen_ms`, and the request's own counters: `req_hits`, `req_hits_profile`, `req_misses`, `req_bytes_upload`,
  `req_bytes_cpu`. With several slots busy at once, a request's counters include the others' lookups (the counters
  are the context's).
- **`exit`**: when the context is freed (normal exit, or the server unloading the model). Adds `layers` as in `load`.
  A killed process writes no `exit` record.

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
| `tools/check_audit.py` | `check_audit.py MODEL.gguf PROFILE.strp [--cache-mib 256]` |
| `tools/monitor.html` | served over HTTP: `monitor.html?server=http://127.0.0.1:8080` |
| `tools/check_all.sh` | every check, CPU then GPU (`--cpu-only` to stop after the CPU ones) |

The tools find the llama.cpp binaries in `$LLAMA_BIN`, else `llama.cpp/build/bin` in this repo
(`scripts/setup.sh`), else `~/projects/llama.cpp/build-cuda/bin`.
