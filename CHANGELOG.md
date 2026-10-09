# Changelog

## 0.1.0 - unreleased

First public version: [Strata](https://github.com/Niko1221/Strata)'s expert tiering for every llama.cpp MoE model.
llama.cpp `de7fa0a` + `patches/llama.cpp-extended-strata-moe.patch`.

**llama.cpp**

- `--moe-cache-profile FILE`: fill the MoE cache at load from a Strata `STRP` profile, hottest experts first.
- `--moe-cache-hybrid`: Strata's hybrid. A static, profile-ranked set of experts in VRAM; the CPU computes the
  others from RAM in the same step; nothing is uploaded during generation. Layers held whole skip the CPU.
- The CPU's `mul_mat_id` and CUDA's one-token `mul_mat_vec_q` skip rows with expert id -1.
- `--moe-cache-audit FILE`: a JSONL audit trail (load, ready, one record per server request, exit).
- `llama_moe_cache_get_stats`, `llama_moe_cache_get_layer_stats`, `llama_moe_cache_get_resident`.
- llama-server: `llamacpp:moe_cache_*` Prometheus series (totals and per layer), and a per-request log line.
- `test-llama-archs`: `--all-host-experts` and `--quant-experts`; cache, profile and hybrid configs with
  bit-identity and first-MoE-layer checks.

**Tools**

- `routing_trace`, `moe_plan.py`, `sim_cache.py`, `bench_cache.py` (interleaved, thread sweeps), `monitor.html`,
  `check_audit.py`, and the check suite (`check_all.sh`).
- `scripts/setup.sh`: clone, patch and build in one step.
