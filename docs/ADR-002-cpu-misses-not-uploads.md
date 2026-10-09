# ADR-002: Compute expert misses on the CPU, not upload them

**Status:** Implemented (`--moe-cache-hybrid`), correct on all 56 MoE architectures; the speed verdict waits on a
large-expert model (see "Implementation and results" at the end)
**Date:** 2026-10-09
**Deciders:** project owner
**Revises:** the step order of [ADR-001](ADR-001-generic-moe-expert-tiering.md)

## Context

ADR-001 extended llama.cpp's MoE cache (`--moe-cache-mib`) with a profile warm start (`--moe-cache-profile`, done).
It left a hybrid mode (Step 2) and heat-based replacement (Step 3) for later. Then we measured a real MoE model:
IBM Granite 3.1 1B-A400M (Q4_K_M; 24 layers, 32 experts, top-8). The box is an RTX 4060 Laptop (8 GB, PCIe x8),
with llama.cpp `extended-strata-moe` at de7fa0a + our patch, CUDA 13.4.

**Decode speed** (`tools/bench_cache.py`: 128 greedy tokens after a 1,500-character prompt, best of 2):

| placement | VRAM for experts | tokens/s | decode hit rate |
|---|---:|---:|---:|
| all on the GPU | 698 MiB | 373.5 | |
| `--cpu-moe` (all experts on the CPU) | 0 | 98.1 | |
| `--n-cpu-moe 22` (last 2 layers' experts on the GPU) | ~64 MiB | 105.7 | |
| cache 64 MiB, cold / from the profile | 64 MiB | 40.9 / 41.3 | 0.00% / 0.05% |
| `--n-cpu-moe 16` | ~230 MiB | 128.8 | |
| cache 256 MiB, cold / from the profile | 256 MiB | 83.9 / 84.9 | 60.9% / 61.4% |
| `--n-cpu-moe 11` | ~380 MiB | **157.8** | |
| cache 384 MiB, cold / from the profile | 384 MiB | 130.7 / 129.8 | 84.8% / 85.9% |

**Hit rates of the policies**, from `tools/sim_cache.py`. Routing was traced on llama.cpp's docs (76k tokens) and
scored token by token on Strata's DETAILS.md (37k tokens, held out). The simulator uses llama.cpp's own slot
arithmetic and LRU.

| cache | layer split | LRU cold (upstream) | LRU + profile (ours) | pinned profile + LRU reserve | static profile, misses on the CPU (Strata) |
|---:|---:|---:|---:|---:|---:|
| 64 MiB | 8.3% | 0.0% | 0.0% | 18.6% | 22.9% |
| 128 MiB | 16.7% | 0.0% | 0.0% | 35.2% | 38.4% |
| 256 MiB | 33.3% | 49.7% | 49.7% | 60.0% | 62.4% |
| 384 MiB | 54.2% | 77.0% | 77.0% | 78.7% | 80.3% |

Four things follow:

1. **Upload-on-miss loses to the plain layer split at every size**, even at an 85% hit rate (130 vs 158 tokens/s).
   A miss costs the expert's bytes over PCIe (measured: 2-3.5 GiB/s of uploads per request), while the layer split's
   misses are computed by the CPU straight from RAM.
2. **The LRU thrashes below one token's sweep.** With fewer slots than (layers in a group) x top-k (34 vs 96 here),
   every expert is evicted before its layer comes round again: 0% hits, 2.4x slower than no cache at all.
3. **The profile warm start barely helps the LRU** (+1%): the first sweeps evict it. It still counts as a hit
   source in the metrics, at 8-13% of lookups.
4. **Per-expert residency is still the right idea.** A frequency-ranked static set holds 2-3x the hits of the layer
   split at the same VRAM (22.9% vs 8.3% at 64 MiB, 62.4% vs 33.3% at 256 MiB). What is wrong is paying for its
   misses with uploads.

## Decision

**Build Strata's hybrid next.** A static, profile-ranked set of experts lives in VRAM and never changes during
decode. Every other expert is computed by the CPU from RAM in the same step, and nothing is uploaded. Pinning in
upload mode is not built.

Until the hybrid lands, the planner warns when a cache would thrash and prints the `--n-cpu-moe` split for the same
VRAM (`tools/moe_plan.py`). It also tells users to measure both (`tools/bench_cache.py`).

## Options considered

### Option A: pinned profile + LRU reserve, upload mode

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low: a pinned flag per slot in `moe_cache_lru`, and `get_slot_map`'s check against the unpinned slots |
| Expected gain | Hit rate 18.6% vs 0% at 64 MiB in the simulator, but each miss is still an upload |
| Risk | Measured upload-mode speed is below the layer split even at 85% hits |

**Pros:** small diff, fixes the thrash. **Cons:** it cannot beat the layer split on this hardware. Rejected.

### Option B: hybrid, misses on the CPU (chosen)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium-high: two expert paths in `build_moe_ffn` and a small CPU op that splits the ids |
| Expected gain | Estimate, not a measurement: the layer split ran at 87% of its CPU-bytes bound (129 vs 148 tokens/s). The static set at 256 MiB leaves 65 of 174 MiB per token to the CPU: ~0.87 x 98 x 174/65 ≈ 230 tokens/s vs 129 for the layer split. Hybrid has to show this |
| Scalability | Every MoE arch, like ADR-001 |

**Pros:** keeps the 2-3x hit-rate advantage of per-expert residency and drops the PCIe cost. This is how Strata
reaches its numbers. **Cons:** CPU and GPU overlap depends on ggml's scheduler, and the summation order changes,
so it is NMSE-gated, not bit-exact.

### Option C: recommend `--n-cpu-moe` and stop

**Pros:** nothing to build; the planner already prints it. **Cons:** gives up a measured 2-3x in hit rate. It stays
the fallback the planner suggests.

### Option D: faster uploads (prefetching the next layer's experts during this layer)

**Pros:** hides PCIe latency. **Cons:** decode routing is only known one layer ahead, it still moves the same bytes,
and at 2-3.5 GiB/s the bytes are the bottleneck, not the latency.

## Trade-off analysis

PCIe bandwidth decides between A/D and B. On this x8 link, uploading a miss costs more than computing it on the CPU.
On a PCIe 5.0 x16 desktop (~4x the bandwidth) upload mode would close part of the gap, but the same expert bytes
still cross the bus. B pays CPU time only for the misses and nothing for the hits. Every user's machine has a CPU
with RAM bandwidth. B is the bigger change, but it is the one the data supports. Large batches (prompt
processing) keep their current path, so B changes nothing there.

## Consequences

- Easier: one placement policy (static per-expert) that the profile, the simulator and the planner already model.
- Harder: two expert paths in `build_moe_ffn`; the metrics must report CPU-computed experts instead of uploads.
- Revisit: on a fast-PCIe desktop, re-run `tools/bench_cache.py`. If upload mode wins there, Option A is cheap.
- Revisit: the estimate in Option B must be replaced by a measurement before this ADR is Accepted.

## Action items

1. [x] Measure (`tools/bench_cache.py`), simulate (`tools/sim_cache.py`), and write down the results (this ADR).
2. [x] Planner: thrash warning, the thrash-free size, and the equivalent `--n-cpu-moe` split.
3. [x] Metrics, logs and the monitor (`llama_moe_cache_get_stats`, `/metrics`, the per-request line,
   `tools/monitor.html`), so the hybrid's effect will be visible the same way.
4. [x] Hybrid in `build_moe_ffn` (`--moe-cache-hybrid`), as built - see below.
5. [x] Gate: `test-llama-archs --all-host-experts`, hybrid configs on all 56 architectures; Granite benchmark.
6. [x] Benchmark a large-expert model on this box: gpt-oss-20b, the same experts as gpt-oss-120b (below).
7. [ ] The same on native Windows, where a GPU-CPU round trip should cost less than under WSL.
8. [x] An audit trail (`--moe-cache-audit`): the setup at load, one record per server request, the counters at
   exit; `tools/check_audit.py` checks that the records add up. Writing it found that llama-server decodes a 2-token
   probe at load (`common_context_can_seq_rm`), outside any request: hence a `ready` record.
9. [ ] gpt-oss-120b (36 layers x 128 experts): the case where the layer split pays nearly as many round trips.

## Implementation and results

**How it works** (`src/llama-moe-cache.cpp`, `src/llama-graph.cpp`, one line in `ggml-cpu.c`):

- The cache is filled from the profile at load and never changes; nothing is uploaded during decoding.
- Layers the cache holds whole take the cache's GPU path with no CPU work, so no round trip at all.
- In every other layer:
  - A CPU op splits each token's selected experts into GPU slots (a stand-in slot for the CPU's experts) and CPU
    expert ids (-1 for the GPU's experts), plus a hit mask.
  - Each (token, expert) pair becomes its own one-expert row. A stand-in can repeat an expert inside a token, and
    CUDA's `mm_ids_helper` keeps one row per expert per token: duplicates left rows unwritten (garbage, NaN), which
    the first version hit on 40 architectures.
  - The CPU's `mul_mat_id` now skips rows with id -1, so the CPU neither reads nor computes the GPU's experts. Before
    this change the masked rows cost a full matrix-vector product each, which on Granite's small experts cost as
    much as real ones.
  - The CPU computes its experts from host memory, the GPU its slots, and the masked results are added on the GPU.
    Biases and per-expert scales always use the real ids.
- The graph is expanded GPU input → CPU (id split + experts) → GPU (masks, cached experts, sum). That is one round
  trip per layer, as with `--cpu-moe`; the default order took two.

**Correctness:**

- `test-llama-archs --all-host-experts` runs two hybrid configs on all 56 MoE architectures: one holding about half
  of each layer's experts, and one holding them all. All pass, including GLM-5.2 (`glm-dsa`), GLM-5.3-Flash
  (`glm5-next`) and gpt-oss.
- Logits of these tiny fixtures hide a wrong expert inside the 1e-4 NMSE, so the test also compares the first MoE
  layer's own output against the cache holding every expert. Correct code measures 1.2e-6 .. 3.4e-6 there, a
  deliberately broken mask 2e-2 .. 1.5; the gate is 1e-4, and it catches the broken mask on all 4 architectures
  tried.
- The fixtures' experts are F16/F32 and their ubatches 2 tokens, so the configs above never took the one-token path
  where CUDA skips the CPU's rows (quantized experts only). `--quant-experts` makes the experts Q8_0, and two
  one-token configs run (a full cache as the reference, then the hybrid), so each comparison is at the same ubatch
  size. The first version compared the one-token hybrid against a 2-token reference: a different ubatch size changes
  the GPU kernels before the layer, which on deepseek4 (9e-3 from the CPU without any cache) gave 7.8e-3. A broken
  skip path (the GPU's ids all -1) is caught on qwen3moe, glm-dsa, gpt-oss and deepseek2 (0.2 .. 0.99).
- Results, 56 fixtures (`build/archs/gpu_matrix.md`, `gpu_matrix_q8.md`):

  | experts | every config passes | ok, upstream | n/a (dense) |
  |---|---:|---:|---:|
  | F16/F32 | 51 | 3 (deepseek4, qwen3vlmoe, smallthinker) | 2 |
  | Q8_0 | 47 | 7 (arctic, deepseek4, gpt-oss, granite, granitemoe, maple, smallthinker) | 2 |

  "ok, upstream": llama.cpp's GPU path differs from its CPU there without the cache too (deepseek4 9e-3, granite
  1.03e-4 against the 1e-4 gate), and no cache config does worse.

**Speed on Granite 1B-A400M** (RTX 4060 Laptop under WSL, 4 CPU threads, 128 tokens):

| config | ms/token | tokens/s | GPU-CPU splits |
|---|---:|---:|---:|
| all on the GPU | 2.69 | 372 | |
| `--cpu-moe` | 9.66 | 103.5 | 50 |
| hybrid 256 MiB (61% hits) | 8.47 | 118.1 | 50 |
| `--n-cpu-moe 16` | 7.33 | 136.4 | 34 |
| hybrid 384 MiB (85% hits) | 7.57 | 132.1 | 50 |
| `--n-cpu-moe 11` | **6.02** | **166.1** | 24 |

What the steps measured, all on Granite:

| hybrid version, 256 MiB, 4 CPU threads | ms/token |
|---|---:|
| one round trip a layer (the first version had two: 12.6 -> 11.6 ms at the default thread count) | 10.39 |
| CPU skips the GPU's rows | 8.87 |
| masks and sum on the GPU | 8.47 |
| same graph with no CPU expert work at all (a timing experiment, wrong output) | 4.6-4.8 |

**What this says.** The hybrid beats `--cpu-moe` but not the layer split on Granite. Every layer that touches the
CPU pays a round trip (~0.2 ms here under WSL), and Granite's experts are so small (0.9 MB) that this costs more than
all the expert bytes the hybrid saves. The layer split wins by removing round trips. The hybrid only shrinks the
bytes. Holding whole layers (`tools/moe_plan.py --whole-layers N`) gives the hybrid the layer split's advantage for
those layers.

The bytes dominate instead when experts are large. gpt-oss reads 4 experts x 13.2 MB per layer per token, about
1.3 ms at 40 GB/s, against the same ~0.2 ms round trip. That is where per-expert residency should beat whole layers,
and the measurement this ADR still needs (action item 6). Native Windows should also have cheaper round trips than
WSL's paravirtualized GPU driver; not measured.

### gpt-oss-20b (MXFP4, 24 layers x 32 experts of 12.6 MiB, top-4)

The routing was traced on 15,678 tokens of llama.cpp's docs and scored on 9,097 held-out tokens. Hit rates from
`tools/sim_cache.py`:

| VRAM for experts | layer split | static per expert (hybrid) | LRU (upstream, uploads) |
|---:|---|---|---|
| 2.5 GB | 25.0%, CPU reads 908 MiB/token | 52.0%, CPU reads 581 MiB/token | 58.3%, uploads 505 MiB/token |
| 3.8 GB | 37.5%, 756 MiB | 67.5%, 393 MiB | 78.0%, 266 MiB |
| 5.0 GB | 50.0%, 605 MiB | 79.8%, 244 MiB | 88.5%, 139 MiB |

The LRU does not thrash here (197 or more slots against 24 x 4 = 96 experts a token).

**A second GPU cost, found and fixed.** The first gpt-oss runs had the hybrid 10-13% behind the layer split. Timing-only
variants split its 32.5 ms per token:

- the CPU's experts cost 10.6 ms;
- the GPU's cost 11.8 ms, 2.5x what its hits should take.

The cause: CUDA's one-token kernel (`mul_mat_vec_q`) reads an expert per row, so the GPU's stand-in rows read full
experts. The GPU read all 4 experts of a token in every layer, whatever the hit rate.

The fix makes that kernel skip rows with id -1 (zero output, no weight read). The hybrid uses -1 on the GPU's side
only where that kernel is certain: one token, quantized experts, CUDA or ROCm. Elsewhere it keeps the stand-ins.
Numerics are checked by KL divergence against `--cpu-moe` with one-token ubatches, so every token takes this path:

| model | hybrid KLD, same top token | upload cache, for scale |
|---|---|---|
| Granite (Q4_K) | 0.0059, 95.8% | 0.0054, 93.7% |
| gpt-oss-20b (MXFP4) | 0.0020, 99.0% | 0.0025, 98.4% |

**Speed after the fix.** The laptop's thermal state moved between runs by up to 25%, so these are interleaved A/B
rounds (tokens/s, 4 CPU threads):

| VRAM for experts | `--n-cpu-moe` | hybrid |
|---:|---|---|
| 3.8 GB | 21.1, 22.6, 20.5 (15) | 22.6, 21.5, 21.2 |
| 5.0 GB | 24.5, 24.3, 23.9 (12) | 26.9, 26.5, 24.0 |

At 4 threads the hybrid is about 6% ahead at 5 GB and level at 3.8 GB. Holding 4 or 8 layers whole beside the
per-expert set measured no better (the runs were too noisy to rank them). Upload mode stayed last at every size (11.5 /
15.5 / 21.2 tokens/s in the run before).

**Correction: thread count.** The earlier "4 threads is as good as 6 or 8" came from a sweep that ran every count at
`-t 4` (the script set a variable it had already read). Rerun at 5 GB, interleaved, 3 rounds:

| | `-t 4` | `-t 6` | `-t 6 --poll 100` |
|---|---|---|---|
| `--n-cpu-moe 12` | 25.2, 24.4, 27.2 (avg 25.6) | 28.2, 30.2, 28.3 (avg 28.9) | avg 28.6 |
| hybrid | 25.0, 26.9, 27.1 (avg 26.3) | 27.7, 28.1, 27.5 (avg 27.8) | avg 27.0 |

Six threads (the i7-13620H's P-cores) is the best count for both, and there the layer split is about 4% ahead. The
layer split gains more from threads because its CPU work per round trip is four experts, the hybrid's about one (78%
hits). Spinning the threads between steps (`--poll 100`) does not help. On gpt-oss-20b the two are level, within
noise.

So with large experts the hybrid matches the best layer split, despite twice the hits. Two costs keep it from the
full byte saving:

- a GPU-CPU round trip in all 24 layers (the layer split has 12);
- serial CPU and GPU expert work.

On gpt-oss-120b the static set's advantage in hits over the layer split should be larger, because 8 GB holds a
smaller share of its experts. That is not measured.

### Fixes from the bug hunt

| # | finding | fix | check |
|---|---|---|---|
| 1 | The tracer could silently drop ubatches when the context lowered `n_ubatch`; the matrix checked only that layers agreed | decode one ubatch at a time (`llama_n_ubatch`) | `arch_matrix.py` now checks routings = tokens x top-k: 53/53 |
| 2 | `--whole-layers` ranked the requested layers' experts together, so a group without room for all of them held few whole (13 asked, 6 held) | one layer at a time, the last first | 13 -> 12 held, 8 -> 7; the planner's count = llama.cpp's log |
| 3 | `--moe-cache-hybrid` / `--moe-cache-profile` without `--moe-cache-mib` were silently ignored | a warning at argument parsing | shown |
| 4 | a stray `ggml_cont` node per layer in upload mode | built only on the hybrid's path | tests pass |
| 5 | the server's per-request snapshot had no null-context guard | guard | - |
| 6 | the hybrid's whole layers synced the GPU per layer to read ids they never use | copy the static slot map without the sync | no measurable change |
| 7 | the tracer ran on llama.cpp's default 4 threads | a thread argument | 15 -> 41 tokens/s on gpt-oss-20b |
| 8 | the GPU's stand-in rows read full experts (above) | CUDA's one-token kernel skips -1 | KLD above; 10-13% behind -> level |
| 9 | `gptoss_bench.sh threads` set `THREADS` after `T` was read, so its sweep ran `-t 4` throughout | set `T` | the correction above |
| 10 | `bench_cache.py` ran each config's rounds back to back (thermal drift favors whichever runs first) and had no thread option | interleaved rounds; `-t` takes a list | the correction above |
| 11 | the architecture matrix never ran the hybrid's one-token GPU path: the fixtures' experts are F16/F32 and their ubatches 2 tokens, and CUDA skips -1 rows only for quantized experts, one token at a time | `test-llama-archs --quant-experts` (Q8_0 experts) and one-token configs, each checked against a full cache at the same ubatch size | see "Correctness" below |
| 12 | `gpu_matrix.py` could excuse a cache-only failure (bit identity `c`, first MoE layer `m`) when llama.cpp without the cache failed too | cache-only failures never pass; other failures must be no worse than without the cache | - |
| 13 | `test-llama-archs`' causal-toggle check decoded an empty batch with one-token ubatches | at least one token | - |
| 14 | `moe_plan.py` printed upload-mode advice ("every miss is uploaded") and no hybrid flags | the hybrid's flags with `--out`; the thrash note only for upload mode | - |

**Known limits:**

- The hybrid serves batches of up to 32 tokens; prompts keep the existing path.
- Outside one-token decoding of quantized experts on CUDA/ROCm, the GPU's stand-in rows still read a full expert
  each. That covers prompts up to 32 tokens, speculative batches, F16/F32 experts, and other GPU backends.
- CPU and GPU expert work run one after the other, not in parallel. Strata overlaps them, and that is the next step
  if the large-expert benchmark shows the CPU time dominating.
