# ADR-002: Compute expert misses on the CPU, not upload them

**Status:** Implemented (`--moe-cache-hybrid`, the mailbox, `--moe-cache-adaptive`), correct on all 56 MoE architectures;
gpt-oss-120b on an 8 GB card runs 69% faster than the best layer split with no profile (see "Implementation and results")
**Date:** 2026-10-09 (adaptive cache: 2026-10-10)
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
9. [x] gpt-oss-120b (36 layers x 128 experts): level with the layer split. Running it found the expert-bias bug, which
   also moved gpt-oss-20b from "level" to +6% (below).
10. [x] OpenAI and Anthropic APIs on the hybrid: `tools/serve.py`, `tools/check_api.py`, `docs/connect.md`
    (llama-server already serves both; the cache changes nothing about the protocols).
11. [x] Overlap the CPU's and the GPU's experts (the mailbox): gpt-oss-120b 17.2 -> 21.3 tokens/s on the same profile,
    gpt-oss-20b 29 -> 36; best layer split to best hybrid, +31% (+59% with a profile from similar text) on 120b.
12. [x] A cache that follows the conversation (`--moe-cache-adaptive`; the profile's text was worth 8 ms a token): see
    "The adaptive cache".
13. [ ] Overlap a prompt batch's expert upload with its compute (prompt speed), see the known limits.

## Implementation and results

**How it works** (`src/llama-moe-cache.cpp`, `src/llama-graph.cpp`, one line in `ggml-cpu.c`):

- The cache is filled from the profile at load; nothing is uploaded during decoding. (It stays as filled unless
  `--moe-cache-adaptive` swaps experts in and out between tokens, see "The adaptive cache" below.)
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

**What this said.** The hybrid beat `--cpu-moe` but not the layer split on Granite. Every layer that touches the
CPU paid a round trip (~0.2 ms here under WSL), and Granite's experts are so small (0.9 MB) that this cost more than
all the expert bytes the hybrid saved. The layer split won by removing round trips. The hybrid only shrank the
bytes. **That was before the mailbox**, which makes the round trip nearly free and runs the GPU's part beside the
CPU's. With it, interleaved, 3 rounds (`--n-cpu-moe 11` is the best layer split; hybrid 384 MiB, 85.6% hits):

| config | threads | ms/token | tokens/s |
|---|---:|---:|---:|
| `--n-cpu-moe 11` | 4 | 6.24 | 160.3 |
| `--n-cpu-moe 11` | 6 | 6.45 | 155.0 |
| hybrid 384 MiB, scheduler's copies (`LLAMA_MOE_CACHE_NO_MAILBOX=1`) | 6 | 8.29 | 120.7 |
| **hybrid 384 MiB, mailbox** | 4 | **4.03** | **247.9** |
| hybrid 384 MiB, mailbox | 6 | 4.10 | 244.1 |

So the round trips, not the experts' size, were what the layer split won on: even Granite gains 55% over the best layer
split. (Holding whole layers, `tools/moe_plan.py --whole-layers N`, is no longer needed to compete.)

The bytes dominate instead when experts are large. gpt-oss reads 4 experts x 13.2 MB per layer per token, about
1.3 ms at 40 GB/s, against the same ~0.2 ms round trip. That is where per-expert residency should beat whole layers
(measured below, gpt-oss-20b and 120b). Native Windows might have cheaper round trips than WSL's paravirtualized GPU
driver; not measured (the mailbox makes the question matter less).

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

At this point the hybrid matched the best layer split despite twice the hits, and the cost was blamed on a GPU-CPU
round trip in all 24 layers (the layer split has 12) and serial CPU and GPU expert work. **That explanation was
wrong; see the next section.**

### Correction: the expert biases (found on gpt-oss-120b)

gpt-oss has a bias per expert. `--cpu-moe` puts the bias tensors in host memory too (its pattern matches
`ffn_*_exps.bias`), and the hybrid built the cache's GPU ops with those host biases. Each `ggml_add_id` with a host
bias is scheduled on the CPU, so the graph left the GPU and came back **four times a layer** (gate, up, down, and the
merged op) instead of once. The scheduler's split count shows it (`GGML_SCHED_DEBUG`, one-token graph, gpt-oss-20b):

| | graph splits |
|---|---:|
| `--cpu-moe` | 50 |
| hybrid, before | 194 |
| hybrid, after | 50 |

The fix: the cache keeps device copies of the layers' host expert biases (a few MiB, out of the budget) and the ops
on the cache's experts use them. `test-llama-archs` passes on all 56 architectures with both F16 and Q8_0 experts
after the change, and the API check passes.

gpt-oss-20b again, the layer split against the hybrid, 6 threads, 4 interleaved rounds (ms per token):

| VRAM for experts | `--n-cpu-moe` | hybrid | |
|---:|---|---|---|
| 3.8 GB | 29.4, 31.6, 28.9, 28.8 (avg 29.7, 33.7 tokens/s; 15) | 28.2, 27.9, 27.8, 27.9 (avg 28.0, 35.8 tokens/s) | +6%, ahead in every round |
| 5.0 GB | 25.7, 26.1, 25.5, 26.0 (avg 25.8, 38.7 tokens/s; 12) | 24.2, 24.5, 24.2, 24.5 (avg 24.3, 41.1 tokens/s) | +6%, ahead in every round |

The "level" result above and the "+6%" at 4 threads before it were both measured with the extra round trips. The
absolute speeds are higher than in the earlier tables because the laptop's state differs between sessions; only
compare numbers inside a table.

### gpt-oss-120b (MXFP4, 36 layers x 128 experts of 12.6 MiB, top-4, 63.4 GB)

The routing was traced on the same 15,678 + 9,097 tokens (18 + 8 minutes on 10 CPU threads). One layer of experts is
1618 MiB, so 2 and 3 layers (`--n-cpu-moe 34` and `33`) match 3240 and 4860 MiB of cache. The simulator at those sizes:

| VRAM for experts | layer split | static per expert (hybrid) | LRU (upstream, uploads) |
|---:|---|---|---|
| 3240 MiB (256 slots) | 5.6% hits, CPU reads 1715 MiB/token | 23.3%, 1392 MiB | 29.4%, uploads 1282 MiB |
| 4860 MiB (384 slots) | 8.3%, 1664 MiB | 30.7%, 1257 MiB | 42.1%, 1052 MiB |

Speed, 6 threads, 4 interleaved rounds, after the bias fix (ms per token per round, average):

| config | rounds | avg | tokens/s |
|---|---|---:|---:|
| `--cpu-moe` | 59.6, 60.7, 59.9, 58.2 | 59.6 | 16.8 |
| `--n-cpu-moe 34` | 56.5, 56.5, 58.3, 58.5 | 57.5 | 17.4 |
| hybrid, 3240 MiB | 56.7, 55.6, 57.9, 57.6 | 56.9 | 17.6 |
| `--n-cpu-moe 33` | 57.0, 53.6, 55.7, 58.0 | 56.1 | 17.8 |
| hybrid, 4860 MiB | 53.2, 55.6, 56.6, 56.5 | 55.5 | 18.0 |

The hybrid is about 1% ahead at both sizes, within the round-to-round noise (2-3 ms): **level**. The static set
serves about 4x the layer split's share of the experts, yet the speed barely moves, because 8 GB holds only 8% of this
model's experts and the CPU reads most of them either way: the best config is 7% faster than `--cpu-moe`. Before
the bias fix the hybrid was 11-28% behind here (9.9 against 11.2 tokens/s at 4860 MiB, 8.5 against 11.8 at 3240).
Why 25% fewer CPU bytes (1257 against 1664 MiB per token) gave only 1% was not known; the next section measures it.

The hit rate measured on the benchmark text (the profile was traced on llama.cpp's docs, the prompt is from
Granite's test file) is lower than the simulator's on held-out docs: 26.3% of decode lookups at 4860 MiB and 18.9% at
3240 MiB, against the simulated 30.7% and 23.3%.

### Where a token's time went

`GGML_SCHED_PROFILE=1` (added for this) splits a generated token's time into the CPU's compute, the GPU's, and the waits
and copies between them. gpt-oss-120b, 6 threads, ms per token:

| | `--cpu-moe` | hybrid, 4860 MiB |
|---|---:|---:|
| CPU computing experts | 41.9 | 33.4 |
| step, all in | 57.4 | 53.5 |
| **step with the CPU's expert work removed** (timing experiment, wrong output) | **12.2** | **18.3** |
| synchronous copies a step | 110 | 182 |

The CPU part did what the cache promised: 8.5 ms less, from 22% fewer expert bytes. The GPU side gave 6 ms back: 12.2 ms
of GPU work and hand-overs without any CPU work in `--cpu-moe`, 18.3 ms in the hybrid. Of the 6 ms, about 2 are 72 more
synchronous copies (27 µs each) and about 4 the GPU computing its hot experts, which it did after the CPU had finished.
Two measurements say what a hand-over should cost (an `nvcc` micro-benchmark, `build/rt.cu`):

| GPU <-> CPU hand-over under WSL2 | per round trip |
|---|---:|
| launch + copy out + sync + copy back (the driver's floor, any sync mode) | 50 µs |
| what the scheduler's path costs here | 130 to 180 µs |
| a flag in page-locked host memory (a GPU kernel and a polling CPU thread) | **2.8 µs** |

The rest of a token is the GPU's dense work (about 1.6 GiB of attention, router and output weights, Q8_0, 7 to 10 ms) and
the CPU's expert reads, which run at 45 GB/s against a measured ceiling of 62 GB/s for 6 threads
(`build/bw.cpp`: streaming reads, 4 KiB or huge pages, pinned or not).

### The mailbox: the GPU and the CPU at once

The scheduler hands a layer over with 2 to 5 synchronous copies and a stream synchronization, and runs its splits one
after the other. The mailbox (CUDA, one token, quantized experts; `ggml/include/ggml-mailbox.h`) replaces both:

- A page-locked, GPU-mapped buffer holds, per layer, a flag, the selected experts, the layer's input and the CPU's result.
- Two GPU ops (`GGML_OP_CUSTOM`, the GPU backend runs those whose function is `ggml_mailbox_cpu_stub`): **send** stores
  the input and the ids, then the flag, stamped with the step's number; **recv** reads the result. A CPU op polls the
  flag (a few µs), splits the ids by the slot map, and the CPU's experts then store their result in the mailbox.
- The GPU looks up its own slots (it keeps a copy of the slot map) and computes the experts it holds right after the
  send, **while the CPU computes the others**. Before, the CPU's part came first and the GPU's after it.
- The scheduler normally waits for the previous split before starting a CPU split that has no inputs (the allocator may
  reuse memory). The CPU op carries a new tensor flag, `GGML_TENSOR_FLAG_NO_SYNC`, that skips this one wait: its split
  reads only the mailbox, whose order the flags keep.
- A stale flag cannot be mistaken for a new one: the flag holds the step number (`begin_step()` before each graph
  compute), the CPU resets it when it has read the ids, and it gives up with an error after 20 s. The GPU never waits for
  the CPU in a kernel (the scheduler enqueues the GPU's next split only when the CPU's is done), so a stuck CPU cannot
  hang the GPU or trip the Windows driver's watchdog.

Checked by:

- `test-llama-archs` with Q8_0 experts, whose one-token hybrid configs now take this path: 56 of 56 architectures ok or
  n/a, and the F16 matrix too;
- the generated text, mailbox against the scheduler's copies (`LLAMA_MOE_CACHE_NO_MAILBOX=1`): identical on 5 of 5
  prompts of up to 160 tokens on gpt-oss-20b, and in `check_audit.py` on every run;
- `check_api.py` (the OpenAI and Anthropic endpoints, 5 requests, through `serve.py`).

### gpt-oss-120b with the mailbox

Interleaved rounds, each config at its best thread count (below), ms per token (average of 3):

| config | threads | ms/token | tokens/s |
|---|---:|---:|---:|
| `--cpu-moe` | 10 | 59.2 | 16.9 |
| `--n-cpu-moe 33` (the best layer split) | 10 | 55.7 | 18.0 |
| hybrid 4860 MiB, profile from llama.cpp's docs (24.1% of decode lookups hit) | 10 | 42.6 | 23.5 (+31%) |
| hybrid 4860 MiB, profile from text like the prompt (44.4% hit) | 10 | 35.0 | **28.5** (+59%) |

At 6 threads, the same hybrid with and without the mailbox (docs profile): 58.2 against 46.9 ms (17.2 against 21.3
tokens/s), while the layer split was 59.8 ms.

And gpt-oss-20b, 5000 MiB of experts, the same method (3 interleaved rounds; the docs profile hit 77.3% of decode
lookups, the profile from the held-out part of the prompt's own text 87.3%):

| config | threads | ms/token | tokens/s |
|---|---:|---:|---:|
| `--n-cpu-moe 12` | 6 / 8 / 10 | 27.7 / 26.1 / 25.2 | 36.1 / 38.3 / **39.6** |
| hybrid, docs profile | 8 | 17.3 | 57.7 (+46%) |
| hybrid, same-text profile | 6 / 8 / 10 | 15.95 / 15.76 / 15.57 | 62.7 / 63.5 / **64.2** (+62%) |

Before the mailbox (and before the bias fix) the same hybrid was level with the layer split here; the "+6%" of the
previous section is the bias fix alone.

**The profile's text is worth 8 ms.** With the docs profile the prompt (ordinary prose) hit 24% of the time; a profile
traced on other text of the same document, 44%: 42.6 against 35.0 ms. The simulator said the same (30.7% against 40.4%
for a profile traced on the test text itself). `tools/make_profile.py` makes one from your own text in one command.

**Threads.** The mailbox changed the best count. Hybrid, same-text profile, tokens/s, from two sweeps of 3 rounds (the
machine's speed differs a little between hours): 6 threads 26.4; 7 27.4; 8 27.6; 9 28.3 and 29.1; 10 28.3 and 29.2;
12 25.0; 14 23.1; 16 15.6. The machine has 6 fast cores with two threads each and 4 slow ones (WSL reports 8 cores of 2
threads, which is wrong); past the fast cores every step waits for the slowest thread. The layer split also gained from
10 threads (60.7 ms at 6, 55.7 at 10). Before the mailbox, on the old path, 7 threads beat 6 by 6% (55.5 against 59.2 ms)
and 12 was 7% worse; CUDA graphs are worth 5% (62.5 ms without); the polling level and pinning the threads to the fast
cores changed nothing.

### Prompt speed and the batch size

A prompt is processed in batches (`-ub`, 512 by default). For each batch llama.cpp uploads the experts of every layer
over PCIe and computes on the GPU: 57 GiB for gpt-oss-120b, about 5.5 s at 10.5 GB/s, whatever the batch holds. Two
checks that this is the bound: without the GPU for the experts (`--no-op-offload`) the prompt runs at 32 tokens/s
whatever the batch; and registering the model's pages as pinned memory (`GGML_CUDA_REGISTER_HOST`) changes nothing
(95 against 95 tokens/s). So a bigger batch is a faster prompt, at the price of its compute buffer, which comes out of
the expert cache. gpt-oss-120b, a 2764-token prompt, 8192 context, 10 threads, the cache `serve.py` fits to the card:

| batch | cache that fits | prompt | generation |
|---:|---:|---:|---:|
| 512 | 4973 MiB | 96 tokens/s | 24.1 tokens/s |
| 2048 | 3863 MiB | 241 | 24.8 |
| 4096 | 2105 MiB | 397 | 21.9 |

(Generation here follows a long prompt, so its hit rates, 16 to 26%, say little; compare the speeds.) 2048 costs nothing,
4096 about 10% of generation for 4x the prompt speed. The buffers add up: weights 1608 + KV 441 + compute 3356 (at
4096) + a 4854 MiB cache is 10,259 MiB, on a card with 8,188. It loaded without an error under WSL2, with the driver
reporting 7,913 MiB used at most: **the driver spills what does not fit**, and the first measurements of the batch
size here (a fixed `--moe-cache-mib 4860` at every batch size) were of a card over-committed that way. `serve.py`
measures the buffers and gives the cache only what is left.

### The adaptive cache

**The problem.** The hybrid's cache was filled at load and stayed that way, so it was only as good as its profile: on
gpt-oss-120b one made from other text hit 24% of the lookups (23.5 tokens/s), one made from the prompt's kind of text 44%
(28.5). Strata has an adaptive tier for this ([ADR-001](ADR-001-generic-moe-expert-tiering.md), step 3). llama.cpp's own
cache adapts too, but by uploading every miss over PCIe, which this ADR showed loses to computing it on the CPU.

**Simulated first** (the held-out gpt-oss-120b trace, 9,097 tokens, 384 slots = 4860 MiB; `sim_adapt.py`, `sim_final.py`):

| policy | hit rate |
|---|---:|
| static, profile from llama.cpp's docs | 30.7% |
| static, profile traced on the test text itself (the best a fixed set can do) | 40.4% |
| adaptive (the design below), from the docs profile | 41.7% |
| adaptive, from an empty cache | 41.6% |
| adaptive, from the test-text profile | 41.7% |

The starting point stops mattering within a few hundred tokens (the three curves are within 3 points in the first 500
tokens and identical after), and the adaptive cache beats even the oracle profile. Swapping one expert a token (41.3%),
four a token (41.6%) or the top by heat every 256 tokens (41.0%) did as well as the design below, which is the simplest
to build: one synchronization and a batch of copies every few dozen tokens, nothing per token. A swap is 13 MB over PCIe,
1.2 ms; the simulated 1,200 to 1,600 swaps over 9,097 tokens are 1.5 to 1.9 s, under 0.5% of the generation time.

**The design** (`src/llama-moe-cache.cpp`):

- *Counting.* The two CPU ops that already split each token's experts into hits and misses (`mailbox_split`, `hybrid_split`)
  add one to the heat of each distinct expert of the layer (a unit that grows by 1/0.998 per generated token, so the old
  counts fade with a half-life of about 350 tokens). No new graph node, no new synchronization.
- *Swapping.* `begin_step()` runs before every graph. After 32 generated tokens (8 while the cache has free slots) it waits
  for the GPU, then fills the free slots with the hottest experts the cache lacks, and swaps the hottest outside for the
  coldest inside while `heat_in > 1.5 * heat_out + 2` units, at most 64 swaps (128 fills) per round. A swap is a blocking
  `ggml_backend_tensor_set` of the expert's bytes from RAM into the evicted expert's slot, then the host slot map and the
  GPU's own copy of it (the mailbox looks experts up there) are updated. Because it happens between two tokens, with the GPU
  idle, there are no copy streams, events or double buffers, and the graph never sees a half-changed bank.
- *What it leaves alone.* Layers the cache holds whole. Prompt batches (they take llama.cpp's path and never reach the
  counting ops). Everything it does is skipped without `--moe-cache-adaptive`.

**Tuning** (gpt-oss-120b, 400 generated tokens from a cold start, one run each, 10 threads; `tune120.sh`):

| every / swaps / when filling, every / fills | start | ms/token | lookups that hit | swaps |
|---|---|---:|---:|---:|
| 64 / 24 / 12 / 128 | docs profile | 38.96 | 31.9% | 144 |
| 64 / 64 / 12 / 128 | docs profile | 38.75 | 38.0% | 260 |
| 32 / 64 / 12 / 128 | docs profile | 41.23 | 34.4% | 319 |
| 32 / 96 / 8 / 128 | docs profile | 37.81 | 41.1% | 366 |
| 16 / 64 / 8 / 128 | docs profile | 38.19 | 41.4% | 381 |
| 64 / 24 / 12 / 128 | empty | 38.35 | 41.4% | 492 |
| **32 / 64 / 8 / 128 (shipped)** | empty | 38.32 | 45.0% | 540 |

Swapping sooner and more raised the hit rate from 32% to 41-45% and left the speed where it was, within the laptop's
run-to-run noise (the 41.23 is one slow run): over 400 tokens the copies cost about what the extra hits save. The shipped
values are the ones with the best hit rate that cost no speed.

**Results.** gpt-oss-120b, 4860 MiB of experts (372 slots), 10 threads, `build/final_bench.sh`: 3 interleaved rounds, each
run 400 generated tokens from a cold start, so the adaptive rows include their own learning (ms per token, then the range
of the 3 runs; the profiles are llama.cpp's docs, "other text", and text from the prompt's own document, "like text"):

| config | ms/token | tokens/s | lookups that hit | swaps |
|---|---:|---:|---:|---:|
| `--cpu-moe` | 68.8 (66.8 to 69.9) | 14.5 | | |
| `--n-cpu-moe 33` (the best layer split) | 62.9 (61.0 to 64.2) | 15.9 | | |
| fixed cache, other-text profile | 44.5 (44.3 to 44.6) | 22.5 | 20.9% | |
| fixed cache, like-text profile | 38.4 (38.3 to 38.5) | 26.0 | 35.3% | |
| **adaptive, no profile** | 37.2 (37.0 to 37.3) | **26.9** | 45.0% | 540 |
| adaptive, other-text profile | 40.3 (40.0 to 40.7) | 24.8 | 34.4% | 319 |
| adaptive, like-text profile | 35.5 (35.3 to 35.6) | **28.2** | 46.3% | 244 |

Against the best layer split that is +69% with no profile and +77% with a like-text profile (the fixed cache: +41% and
+64%). One 1,500-token run of each (tokens/s, hit rate, swaps): fixed other-text 20.7, 17.7%; adaptive no profile 27.9,
47.1%, 643; adaptive other-text 23.2, 30.1%, 642; adaptive like-text 28.8, 49.3%, 310. The same on gpt-oss-20b (5000 MiB
= 395 slots, 51% of its experts), 400 tokens, 3 rounds: `--cpu-moe` 22.8, `--n-cpu-moe 12` 38.7, fixed 60.5 and 62.0
(other-text, like-text profile; 81% and 84% of lookups hit), adaptive 54.6 (no profile), 57.0 and 60.5 (82%, 82%, 86%).
Single runs, tokens/s, at 1,500 and 3,000 tokens: fixed other-text 53.6 and 52.4, fixed like-text 55.2 and 51.4, adaptive
no profile 56.5 and 56.1, other-text 58.6 and 58.8, like-text 60.7 and 65.2 (the adaptive hit rates stay between 79% and
92%; the fixed ones fall from 81-84% to 70-73% as the generated text drifts away from their profile).

What this says:

- On the 120b the adaptive cache with **no profile** beats a fixed cache with a profile made from the right text, and it
  keeps improving (27.9 tokens/s at 1,500 tokens). Its extra hits are worth more than the copies cost because the card
  holds only 8% of the experts, so every hit is a CPU read saved and the fixed set is far from the best one.
- On the 20b, whose cache already holds 51% of the experts, a fixed profile is near the limit, and the adaptive cache's
  learning is a cost: 477 swaps at 1.2 ms are 0.57 s of a 400-token reply (6.7 s), 2 to 12% of the speed. It pays for
  itself before 1,500 tokens.
- **A profile made from other text starts the adaptive cache worse than none** (the 120b: 24.8 tokens/s and 34% of the
  lookups hit, against 26.9 and 45%; the 1,500-token run says the same). It fills the cache with experts the conversation
  does not use. Those the conversation happens to hit early count as proven, and the margin protects them from the
  candidates that are better. On the 20b the same start is ahead of none. Two fixes were tried and changed nothing (the
  hit rates came out identical): running the next round sooner after one that used all its swaps, and replacing a resident
  nobody has selected like a free slot. Not pursued; the cause is unconfirmed, because every run generates its own text and
  the texts drift apart, so a replay of one token stream through each config would be the next measurement.
- The hit rates of different configs are therefore indicative, and the speeds are what a user gets.

What `serve.py` does from this: it always turns the adaptive cache on, with a profile only if you give one.

**A second model: Qwen3-Coder-Next 80B-A3B, Q6_K** (arch `qwen3next`: 48 layers, 512 experts, 3B active; 2,991 MiB of cache at a
4,096 context, `-ub 4096`; 3 interleaved rounds of 400 tokens, 10 threads, no profile). ms/token, tokens/s: `--cpu-moe` 59.8 (16.7),
`--n-cpu-moe 46` 56.1 (17.8), adaptive 32.3 (30.9; 52% of lookups hit, 1,707 swaps), so +74% over the layer split; one 1,500-token
run 33.9 ms (29.5 tokens/s, 47%, 2,261 swaps). `tools/code_eval.py` through `serve.py`: 6 of 6 tasks pass; the harder `--suite hard` (10 tasks, temperature 0, 1,500-token limit)
passes 8 of 10 (dijkstra, topological sort, glob matching, the bsearch fix, text justification, rate limiter, CSV line, retry pass).
The two failures are real code flaws, not format slips: `num_islands` is a recursive DFS (`RecursionError` on 300 x 300 although
the prompt says grids reach 500 x 500), and `evaluate` handles unary minus with a stack hack that is wrong whenever a leading minus
meets `+`/`-`/`*` (`-2+3` -> -5). The calculator first *passed* because the asserts only had `-2*-3`, where the bug cancels; the
asserts were then strengthened and the stored answers re-graded. The context matters
for the fit: the default 32,768 gave the 94-layer Qwen3-235B a 6 GB KV cache and no room for experts, so a model run for a speed
test should pass `--ctx`. Qwen3-235B-A22B (Q2_K, 85.7 GB) was downloaded and then deleted by a faulty script of mine before any
test ran, so there are no numbers for it.

**Where a token's time goes on Qwen3-Coder-Next, and what did and did not help** (Q6_K, 4,603 MiB of cache at `-ub 1024`,
10 threads, `GGML_SCHED_PROFILE` and a counter of the CPU's waits for the mailbox):

| part of a token | ms | bound by |
|---|---:|---|
| the CPU waits for the GPU's attention and router (248.7 us x 48 layers) | 11.9 | GPU memory: 1,845 MiB of weights a token, 7.5 ms at 256 GB/s |
| the CPU computes the experts the cache lacks (457 MiB a token, 34 GB/s) | 13.9 | the RAM and the per-op overhead (below) |
| adaptive swaps, first 400 tokens (2,381 x 0.47 ms) | 2.8 | blocking copies from pageable memory |
| the GPU's combine, sampling, the rest | ~3.5 | |

- *A cache sized for generation*: `-ub 1024` instead of 4096 leaves 4,603 MiB for experts instead of 3,001; 61% of lookups hit
  instead of 49%, 31.7 instead of 34.5 ms. Prompts get slower, so it is `serve.py --favor generation`, not the default.
- *Staged swaps*: the prefill and the swaps copy experts through a 64 MiB pinned buffer, queued on the device's stream while
  the next expert is copied in: 0.47 -> 0.29 ms a swap; 30.0 to 30.6 ms/token over three 400-token runs (33 tokens/s).
- *Guessing the experts before the attention* (removed): the layer's router applied to its input before the attention
  picks 83.5% of the experts the router really picks (94.6% in a guess of 16; the previous token's: 39.2%;
  `tools/route_predict.py` on 768 tokens of code). Built: the GPU sent each layer's input ahead of its attention, the CPU
  guessed with an int8 copy of the router (40 us) and read the guessed experts the cache lacked into its caches until the real
  ids came; 88% of the CPU's experts were read ahead. It saved nothing measurable (37 to 41 ms against 37 to 39, 300-token
  runs, 2 rounds), and an early version, whose readers did not stop, cost 8 to 14 ms. The window is ~140 us a layer, room for
  half of a layer's cold experts, and the experts' compute did not get faster with them in the cache: the CPU part is not only
  RAM-bound. Removed, as code without a gain.
- *Chunked CPU experts*: per op kind (`GGML_CPU_OP_PROFILE`), a token's CPU time was the expert products, 144 of them at
  115-120 us, and the wait for the GPU; barriers cost little (the activation and its barrier: 4.7 us). For one token, each
  expert's rows went to the threads in one slice each, so every expert waited for the slowest thread, and four of the ten run
  on efficiency cores. Chunks of 32 rows taken as they come: 92-96 us an op, 36.0 -> 33.7 ms a token (3 interleaved rounds,
  300 tokens); over 1,200 tokens 26.85 ms, **37.2 tokens/s**, the swaps down to 0.96 ms a token from 3.0 in the first 300.
- *CUDA graphs*: off, launching the GPU's work costs 16-19 ms a token (47 against 34 ms); on, ~45 us a layer, the launch
  latency of WSL2. One launch a token instead of 49 would need the GPU to wait for the CPU inside its graph; not done.
- *The CPU kernels* are not the limit: one-token matrix-vector products run at 46 GB/s for Q6_K, Q8_0, Q5_K and MXFP4 and 58 GB/s for
  Q4_K (`test-backend-ops perf`, 10 threads), the RAM's speed. The experts in the model reach 34 GB/s: ~100 us a layer go to
  the per-op overhead of seven small ops and their thread barriers. Thread counts 6 to 12, `--poll 100` and pinning to the
  P-cores changed nothing (31.1 to 32.6 ms). IQ3_XXS is the exception: 21 GB/s, limited by decoding, which is why Strata's
  model is slow here.

**Strata's own model: Qwen3.8-Flash-Next, GSQ-RCO IQ3_XXS** (125B, 6B active, 48 layers x 512 experts, 75.8 GB in two files read
in place from `C:\Strata-main`, loaded with `--load-mode none` because mmap over `/mnt/c` did not finish in 15 minutes). Stock
llama.cpp with this patch loads and runs the file. One server, adaptive cache, no profile, context 4096, `-ub 2048`: the weights,
compute buffer and KV leave only 1,512 MiB for experts (about 2.6% of them). Three requests in a row, ms/token and tokens/s:
400 tokens 116 (8.6), 400 tokens 105 (9.5), 1,500 tokens 88 (11.4), per-request hit rates 25% to 38%. That is under half of the
20-25 tokens/s Strata gives on this machine (and its README's 53-79 on an RTX 5070). Likely causes, not isolated: the small cache (a
smaller `-ub` would leave about 3.5 GB), the CPU cost of the IQ3_XXS format, and Strata's model-specific kernels. The coding check
passed 5 of 6 tasks; the sixth ran into the 700-token limit before it wrote code (this model answers at length).

**Checked by:**

- `test-llama-archs`, a "MoE hybrid, adaptive" config: tokens are decoded in chunks of 8, between them the cache swaps in
  the experts the chunk selected (eagerly, whatever the heat), and the logits must stay the CPU's, bit for bit the same
  after a save and load, with the first MoE layer matching the full cache. It fails if a run that had misses swapped
  nothing. 56 of 56 architectures with Q8_0 experts (the mailbox) and with F16 (the scheduler's copies).
- Deliberate breaks of the swap code, each run on gpt-oss, qwen3moe, deepseek2 and granitemoe with Q8_0 and with F16
  experts, then restored (`build/teeth_adaptive.sh`):
  - the expert's bytes never copied into the slot: **fails 8 of 8** (the first MoE layer is off by 0.1 to 1.1);
  - the GPU's own slot map not refreshed after a swap: **fails 4 of 4 with Q8_0**, where the mailbox looks experts up in
    it; the F16 runs have no such copy and pass, as they should;
  - the host's slot map not recording the swapped-in expert (the logits stay right, the cache quietly shrinks): this
    passed 8 of 8, until the test counted the experts the cache holds before and after; now **it fails 8 of 8**
    ("63 experts held before and 0 after").
- `check_audit.py` on Granite, no profile: the cache starts empty, 295 swaps fill its 278 slots, and 62% of the lookups
  hit within 160 tokens (an empty hybrid cache without `--moe-cache-adaptive` hits 0%).
- `check_api.py`, through `serve.py`, which now starts the server with the adaptive cache.

**Not done:** the prompt's own routing is not counted (big batches never reach the counting ops), so a reply starts from
the cache the last reply left; counting it would tune the cache to the prompt before the first token. Only one GPU was
tested. A swap stalls the GPU for its copy, 1.2 ms for gpt-oss-120b's experts and up to 64 of them in a round; overlapping
the copies with the next token's CPU work would hide that, for a few percent of the first reply.

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
| 15 | the cache's GPU ops used the experts' host-memory biases: 4 GPU-CPU round trips a layer on gpt-oss instead of 1 (194 graph splits instead of 50) | the cache keeps device copies of the biases | splits 194 -> 50; gpt-oss-20b +6% over the layer split (was level); see "Correction: the expert biases" |
| 16 | `tools/build_cuda.sh` left llama-completion and llama-perplexity out of its default targets, and a rebuild of some tools after a `common.h` change left a stale `llama-cli` that crashed at argument parsing | the default targets are `setup.sh`'s | llama-cli audit check |
| 17 | `gpu_matrix.py` waited 30 minutes per architecture, then crashed instead of reporting | 300 s per architecture, a timeout is a FAIL line | - |
| 18 | the GPU's hot experts ran after the CPU's, and each layer's hand-over cost 130-180 µs in 5 synchronous copies and a stream sync | the mailbox: GPU and CPU work at once, hand-overs of 3 µs | 56/56 both matrices, identical text; gpt-oss-120b 17.2 -> 21.3 tokens/s, 20b 29 -> 36 |
| 19 | a fixed `--moe-cache-mib` with a bigger batch size over-committed the card; the driver spills what does not fit instead of failing | `serve.py` measures the buffers and gives the cache what is left | 10,259 MiB of buffers on an 8,188 MiB card, loaded without an error |
| 20 | WSL shuts its VM down a minute after the last program exits, emptying the file cache: the next run read 60 GB from disk, and the first run of every benchmark was cold | documented; benchmarks keep a program running | the guide |
| 21 | the launcher's measurement ran `llama-completion` with an empty prompt, which it refuses | a one-token prompt | `check_api.py` through `serve.py` |
| 22 | the first adaptive test config failed four architectures that were not broken: it decoded the diffusion models (`llada-moe`, `rnd1`) in chunks of 8 tokens, which changes what a non-causal model attends to, and it demanded swaps from fixtures that never miss (`ernie4_5-moe`: its routing stays inside the cache; `refact`: the cache holds every layer whole) | the encoders are decoded whole and only get the flag; swaps are required only when the run had misses | 56/56 with Q8_0 and with F16 experts |

**Known limits:**

- The hybrid serves batches of up to 32 tokens; prompts keep the existing path.
- Outside one-token decoding of quantized experts on CUDA/ROCm, the GPU's stand-in rows still read a full expert
  each. That covers prompts up to 32 tokens, speculative batches, F16/F32 experts, and other GPU backends.
- The mailbox, and with it the overlap of the GPU's and the CPU's experts, is NVIDIA CUDA, one token at a time. ROCm and
  batches of 2 to 32 tokens keep the scheduler's copies (the code is there to extend: the ops are in `ggml-cuda`).
- A prompt's upload of the experts and its compute run one after the other. Overlapping them (a second staging buffer,
  1.6 GiB for gpt-oss-120b, VRAM the expert cache would lose) could take a 4096-token batch from about 400 to 700 tokens/s.
- Without `--moe-cache-adaptive` the cache's contents are fixed at load, and a profile from other text hits half as often
  as one from the same kind of text (24% against 44%, 8 ms a token). With it, the cache follows the conversation: see
  "The adaptive cache" for what it does not do (count the prompt's routing, overlap its copies).
