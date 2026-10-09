# Contributing

Thanks for helping. The most useful things right now:

1. **Benchmarks from other hardware**: desktop GPUs, PCIe 5.0, 12-24 GB VRAM, native Windows, AMD. Use the
   "Benchmark report" issue template.
2. **Bug reports with an audit log** (`--moe-cache-audit`): it names the exact setup.
3. Code: see below.

## Layout

| path | |
|---|---|
| `patches/llama.cpp-strata-moe.patch` | every llama.cpp change, against the commit in `patches/BASE` |
| `scripts/setup.sh` | clones llama.cpp, applies the patch, builds everything |
| `tools/` | the tracer (C++), the planner, simulator, benchmark and checks (Python), the monitor (HTML) |
| `docs/` | the guide, the reference, and the design records (ADRs) |

## Working on the llama.cpp side

`scripts/setup.sh` leaves `llama.cpp/` on a branch `strata-moe` with the patch applied but not committed. Edit there,
rebuild (`cmake --build llama.cpp/build -j`), and when done regenerate the patch:

```bash
git -C llama.cpp diff "$(cat patches/BASE)" > patches/llama.cpp-strata-moe.patch
```

To move to a newer llama.cpp: check out the new commit in `llama.cpp/`, `git apply --3way` the patch, resolve the
conflicts, put the commit hash in `patches/BASE`, regenerate the patch, and run every check.

Keep the patch in llama.cpp's style: the surrounding code's naming, comment density and idioms. Every MoE
architecture goes through `build_moe_ffn` and `llama_moe_cache`, so a change there covers (and can break) all 53.

## Checks

```bash
bash tools/check_all.sh            # CPU checks, then GPU checks (CUDA)
bash tools/check_all.sh --cpu-only
```

| check | what it guards |
|---|---|
| `tools/test_moe_plan.py` | the planner on synthetic GGUFs; its profiles = what llama.cpp's reader reads |
| `tools/e2e_tiny.sh` | trace -> plan -> profile on a tiny model |
| `tools/arch_matrix.py` | the tracer and planner on every MoE architecture's fixture |
| `tools/check_slots.py` | the planner's slot counts = llama.cpp's |
| `tools/gpu_matrix.py` | llama.cpp's `test-llama-archs` with every cache config on every architecture (GPU) |
| `tools/gpu_matrix.py --quant-experts` | the same with Q8_0 experts: the hybrid's one-token GPU path |
| `tools/check_audit.py` | the audit log adds up (needs a real model) |

A new check is only trusted once it has been seen to fail: break the code on purpose (a wrong mask, a wrong slot),
run the check, see it fail, restore. ADR-002 lists the deliberate breaks used so far.

## Benchmarks

- **Interleave.** Run round 1 of every config, then round 2, and so on (`tools/bench_cache.py` does). A laptop's
  speed moves with its temperature by more than the configs differ, and so does a sequential sweep's order.
- **Compare at each config's best thread count**; report the counts.
- **Report the hardware**, the text the profile was traced on, and the text the benchmark ran on.
- Numbers go into ADR-002 (or a new ADR) with how they were taken.

## Pull requests

Small and focused. Say what changed and why, paste the check summary, and update the docs that describe the
behavior you changed. The PR template has the list.
