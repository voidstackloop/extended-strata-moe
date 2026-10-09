## What and why

<!-- What this changes, and the problem or measurement behind it. -->

## Checks

- [ ] `bash tools/check_all.sh` (or `--cpu-only` without a GPU): paste the summary lines
- [ ] llama.cpp changes: `patches/llama.cpp-strata-moe.patch` regenerated (see CONTRIBUTING.md)
- [ ] a new check was seen to fail on a deliberately broken version
- [ ] speed claims: interleaved rounds (`tools/bench_cache.py`), with the hardware and thread counts
- [ ] docs updated (README, docs/reference.md, docs/guide.md) where behavior or flags changed
