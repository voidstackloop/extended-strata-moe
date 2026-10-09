# ADR-001: Strata's expert tiering for any MoE model

**Status:** Accepted for Step 1 (done); Steps 2-3 reordered by [ADR-002](ADR-002-cpu-misses-not-uploads.md) after
measurement: upload-on-miss loses to the layer split, so the hybrid comes next and pinning is dropped
**Date:** 2026-10-08
**Deciders:** project owner

## Context

[Strata](https://github.com/Niko1221/Strata) runs a 125B MoE model (Qwen3.8-Flash-Next: 48 layers x 512 experts,
top-10) on a 12 GB gaming GPU. Its ideas:

1. **A profiled expert tier in VRAM.** `(layer, expert)` pairs are ranked by measured routing frequency, and the
   hottest fill whatever VRAM the dense weights leave. Strata measures a 0.64 held-out hit rate with about 17% of the
   experts resident.
2. **Every expert stays in pinned RAM, and the CPU computes the misses in place** while the GPU computes the hits.
   A miss never waits for PCIe.
3. **Adaptive swapping**: the tier counts routing heat during a chat and swaps hot experts in, off the critical path.
4. **Speculative decoding** from the model's MTP head, plus prompt lookup (1.6-1.8x).
5. **On the prompt path, the next layer's experts stream to the GPU** while the current layer's attention runs.

**Strata's engine does not generalize.** It is built for one architecture on purpose. `ModelGeometry`
(`include/strata/core/layout.hpp`) holds the dimensions as constants (`n_embd = 2560`, `n_expert = 512`,
`n_ff = 640`). The CPU expert blob has fixed offsets that refuse any other geometry (`expert_source.hpp`). The
mixers (Gated DeltaNet, QSA attention, hyper-connections, the PLE n-gram table) are hand-written CUDA/HIP kernels,
about 4 MB of code, for that model only.

**llama.cpp now has part of idea 1.** Upstream (checked at `de7fa0a`, 2026-10-08) ships
`--moe-cache-mib` (`src/llama-moe-cache.cpp`): an LRU of experts in VRAM for layers whose experts are kept in host
memory. In `build_moe_ffn`, the slot map (`get_rows(slot_map, ids)`) points `mul_mat_id` at the cache. The
scheduler's copy callback reads the selected ids on the host and **uploads every miss over PCIe**, evicting the
least recently used expert. The GPU then computes all k experts. It serves ubatches of up to 32 tokens, and larger
batches only read from it. It **starts empty** and has **no notion of frequency**.

The gap between upstream and Strata:

| | upstream `--moe-cache-mib` | Strata |
|---|---|---|
| start | empty, warms up by LRU | filled from a frequency profile |
| miss | upload the expert over PCIe, then compute on the GPU | compute on the CPU, in parallel with the GPU |
| replacement | LRU, every miss is admitted | heat counts, swaps off the critical path |

**Goal:** bring ideas 1-3, and later 4-5, to the MoE models llama.cpp loads: Qwen3-30B-A3B, Qwen3-Next-80B-A3B,
GLM-4.5-Air, gpt-oss, DeepSeek-V2-Lite and others.

**Constraints:** the dev box is an RTX 4060 Laptop (8 GB) with 46 GB of RAM under WSL, and no CUDA toolkit yet.
Strata's users have 12-24 GB of VRAM and 32-96 GB of RAM. The baselines to beat are llama.cpp's
`--n-cpu-moe N` (with the best N) and upstream's `--moe-cache-mib`.

## Decision

**Extend upstream llama.cpp's MoE cache with Strata's three differences:** a profile warm start, CPU compute for
misses, and heat-based replacement. Do not port Strata's engine, and do not build a second, parallel expert tier.
All MoE architectures in llama.cpp go through `build_moe_ffn` and this one cache, so each change covers every model.

Keep Strata's file formats, so tools and profiles move between the two projects: the `STRP` profile and the
`--dump-routing` trace.

**Measure before each step.** These ideas pay only when routing is skewed. Mixtral (8 experts, top-2) has no "hot
few", and a uniform router gives a hit rate equal to the resident fraction, which `--n-cpu-moe` already gets.

## Options considered

### Option A: generalize Strata's engine

| Dimension | Assessment |
|-----------|------------|
| Complexity | Very high: mixer kernels for each architecture (MLA, sliding window, attention sinks...) on CUDA, HIP and SYCL |
| Cost | Months per architecture family |
| Scalability | Every new model is new kernel work |

**Pros:** Strata's tuned kernels and its MTP path come along. The best ceiling for a single architecture.
**Cons:** It rebuilds llama.cpp one architecture at a time, and upstream Strata refuses other geometries by design.

### Option B: our own hot tier in `build_moe_ffn` (this ADR's first draft, superseded)

**Cons:** It duplicates what upstream's cache already does (banks, slot maps, the copy-callback sync, the
large-batch reuse). Found after the first draft, once the upstream source was read.

### Option C: extend upstream's `llama_moe_cache` (chosen)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low for the warm start (done, ~100 lines); medium for CPU misses |
| Cost | Days for the warm start, then weeks for CPU misses |
| Scalability | Every MoE architecture llama.cpp supports, now and later |

**Pros:** Small diffs on code upstream maintains, and each step can be sent upstream as its own PR.
**Cons:** A fork to rebase until it is merged. How far CPU and GPU work overlap depends on ggml's scheduler.

### Option D: kTransformers

**Pros:** It already does hybrid CPU/GPU MoE for DeepSeek and Qwen MoE.
**Cons:** PyTorch-based and not GGUF-native, with per-model rules. It is a reference point for benchmarks, not a base.

## Design

### Step 1: profile warm start (`--moe-cache-profile FILE`), coded

At load time, each cache group (layers on one device that share an expert layout) takes the hottest pairs of its
own layers from the profile, as many as it has slots. Each one is uploaded into its slot and its slot-map entry is
set. The i-th hottest takes slot `n_slots-1-i`. The LRU list runs from slot 0 (least recent) to the last slot, so
the coldest prefilled experts are evicted first. A missing file, or a file made for another model, logs a warning
and the cache starts empty, as before.

The profile comes from `tools/moe_plan.py` (traces from `tools/routing_trace`).

**The planner mirrors the cache's own arithmetic**, so the slots it reports are the ones llama.cpp allocates:
- **What goes in RAM:** the `--cpu-moe` regex (`\.ffn_(up|down|gate|gate_up)_(ch|)exps`) decides it, so expert
  biases, scales and GroveMoE's chunk experts count as RAM. The `token_embd` input layer also stays in RAM.
- **What the cache holds:** only the four `ffn_*_exps.weight` tensors, in the order llama.cpp uses.
- **Groups and budget:** layers with the same expert types and shapes form one group. Each group gets a share of the
  budget proportional to its host bytes, computed in `double` as llama.cpp does.
- **Bank sizes:** every bank gets one spare slot and is padded to 128 bytes.
- **Minimum size:** a group with fewer slots than experts per token is not cached. If no group is cached, llama.cpp
  refuses the context, and the planner reports the smallest size that works.
- **Pinned placement:** the printed command includes `-fit off`, because llama.cpp otherwise re-fits unset
  parameters to free memory without knowing about the cache.

**Model coverage, checked.** Every MoE graph in llama.cpp reaches `build_moe_ffn`, either directly (59 model files)
or through a shared graph. `tools/arch_matrix.py` runs the tracer and planner on llama.cpp's own tiny model of each
architecture. Of 56 architectures, 53 trace every expert layer for every token, including:
- GLM-5.2 (`glm-dsa`, `GlmMoeDsaForCausalLM`)
- GLM-5.3-Flash (`glm5-next`)
- gpt-oss-20b/120b (`gpt-oss`)
- DeepSeek V4 (`deepseek4`), Kimi K3 (`kimi-k3`), MiniMax M3 (`minimax-m3`)

The other three have nothing to cache: `ernie4_5` and `paddleocr` are dense, and `refact` loads expert tensors but
runs a dense FFN.

GroveMoE runs two routers per layer under one tensor name. The tracer keeps the first, which routes the experts the
cache holds.

### Step 2: CPU computes the misses (hybrid mode)

The copy callback already synchronizes and reads the selected ids on the host. In hybrid mode it does not upload
the misses. Instead it fills, per token, a hot id list (the slot, or a slot this token already uses, with weight 0)
and a cold id list (the expert, or an expert this token already sends to the CPU, with weight 0), plus the weight
mask. Then `build_moe_ffn` runs two `mul_mat_id`: the cache banks on the GPU and the host experts on the CPU.
The two are masked and added. Pointing the zero-weight rows at experts that are already used adds rows, not expert
reads, because ggml's CPU `mul_mat_id` reads only experts that have rows.

There is one known ceiling. When every one of a token's experts is a hit, the CPU side still has to name some
expert, which costs one expert read per layer. A skip flag is the upgrade.

The hypothesis to measure: a miss costs `expert_bytes` over PCIe in upstream's design (about 12 GB/s on this
laptop's x8 link, about 25 GB/s on x16), against the same bytes read from RAM by the CPU (40-60 GB/s), in
parallel with the GPU's hits.

### Step 3: heat-based replacement

Count routing heat per pair. Admit a miss only when its heat beats the coldest resident expert, and upload it
between tokens, not inside the layer. Save the learned ranking back as a profile (Strata's `--expert-profile-save`).

## Phases

| Phase | What | Exit criterion |
|-------|------|----------------|
| **P0** | Planner (done), tracer (done, builds), profile warm start (done, not yet run on a GPU) | On 2-3 models: hit rate and decode t/s of the cold vs. warm cache vs. the best `--n-cpu-moe N`, on the same box and prompts |
| **P1** | Hybrid mode (Step 2) | Same output as stock llama.cpp within a KL gate; decode t/s beats the cold and warm upload modes |
| **P2** | Heat-based replacement + learned-profile save (Step 3) | The hit rate rises over a chat with no upload stalls |
| **P3** | MTP / draft speculation where llama.cpp supports it for the model; prompt-path streaming | Prompt t/s and speculation acceptance |

## Consequences

- Easier: any MoE model llama.cpp loads gets each step with no new kernels, and profiles move between Strata and here.
- Easier: each step is a small, separately upstreamable change.
- Harder: rebasing on a fast-moving upstream until the changes are merged.
- Revisit: if the scheduler serializes the CPU and GPU expert work in Step 2, the CPU side runs inside a custom op on
  a thread pool during the GPU kernel (Strata's design).
- Revisit: coarse-expert models (Mixtral) stay out of scope unless their traces show skew.

## Action items

1. [x] `tools/moe_plan.py` + self-test: any MoE GGUF (split files too) -> slots as llama.cpp computes them, the `STRP`
   profile and the flags.
2. [x] `tools/routing_trace`: llama.cpp eval-callback tracer (CPU build).
3. [x] `--moe-cache-profile` on the `extended-strata-moe` branch of llama.cpp + a cross-language test of the profile reader.
4. [x] Tracer + planner on all 56 MoE architectures (`tools/arch_matrix.py`).
5. [x] CUDA 13.4 toolkit in WSL, user-local (`~/cuda-13.4`, NVIDIA redist archives, no root).
6. [ ] GPU: `test-llama-archs --all-host-experts` (the cache with and without a profile against the CPU, every MoE
   architecture) and `tools/check_slots.py` (the planner's slots against llama.cpp's).
7. [ ] Pick the P0 models and download them (needs the owner's OK). gpt-oss-20b is the same architecture as
   gpt-oss-120b and fits this box.
8. [ ] P0 measurement: cold vs. warm vs. `--n-cpu-moe`; held-out hit-rate curves.
9. [ ] P1 hybrid mode.
