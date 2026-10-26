# One-token routed-expert matvec on this PC (2026-10-26)

Tool: `bench.c` (build with `build.bat`, run with `run.bat <threads>`), CPU backend of `C:\Projects\llama.cpp-strata\build-clang`
(AVX2 + AVX-VNNI, no AVX-512), i7-13620H, DDR5-5200, 10 threads. 10 of 128 experts per op, ids rotated so the weights stream from RAM.
Shapes are k448-like guesses: gate/up k=2048 m=1024, down k=1024 m=2048. A 78 GB download was running in the background (light CPU use).

| type | gate/up Gweights/s (GB/s) | down Gweights/s (GB/s) | note |
|---|---|---|---|
| **iq2_s** (k448 gate/up) | 96.7 (31.0) | 90.4 (29.0) | compute-bound: about half of what RAM allows |
| **iq4_nl** (k448 down) | 93.4 (52.5) | 101.2 (57.0) | close to RAM speed |
| iq3_xxs | 61.4 (23.5) | 70.3 (26.9) | the slow one (Strata's model type) |
| q2_K | 183.1 (60.1) | 174.7 (57.3) | what a RAM-speed 2-bit kernel looks like |
| q4_K | 115.9 (65.2) | 118.8 (66.8) | RAM ceiling is about 65 GB/s |
| q4_0 | 112.7 (63.4) | 114.6 (64.5) | |
| q6_K | 71.7 (58.8) | 63.0 (51.7) | |

Reading: with k448's types the CPU expert work is about 95 Gweights/s. An IQ2_S kernel at RAM speed would be about 190 Gweights/s.
Estimate (not measured on the model): 2.36 B routed weights per token / 95 G = about 25 ms per token if every expert ran on the CPU.

## Server baseline on the model (Minecraft closed, Overwolf on)
no MTP 17.5 tok/s; MTP 1 draft 20.3-20.7; MTP 2 drafts 21-22 (acceptance 93%); GPU cache 645 MiB, hit rate about 14%. With Minecraft running: 11.7 no-MTP. Bigger cache + MTP draft fails to load (bug).

## All-VRAM test
no MTP, cache 3200 MiB, margin 250, ubatch 512: 20.1-22.5 tok/s, hit 52%, VRAM 7947/8188. MTP2 + cache 1800 (margin 700): 21.4-22.4, hit 29%. Cache 2400+ with MTP: draft fails to load (VRAM). They do not stack.

## Cache/draft, long prompts, kernel (2026-10-26)
Draft on CPU + cache 3000: 21.6 tok/s (no gain). Prefill: 5,198 tokens at 164 tok/s; repeat prompt 0.2 s. Decode on prose explanation 14-16 tok/s (draft acceptance 65% vs 93% for code). IQ2_S kernel already AVX2 with table gathers; est. upside under 10%, not written.
