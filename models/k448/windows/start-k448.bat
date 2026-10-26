@echo off
rem Tessera-125B-MoE "k448" (Qwen3.8-Flash-Next, 448 experts, IQ2_S/IQ4_NL) on the RTX 4060 Laptop (8 GB) + i7-13620H, native Windows, extended-strata-moe hybrid.
rem OpenAI API: http://127.0.0.1:8090/v1   Anthropic API (Claude Code): http://127.0.0.1:8090   model name: k448
rem Tuned 2026-10-26 (C:\Projects\k448-engine\bench\RESULTS.md): 128K context with q8_0 KV (1836 MiB), expert cache 950 MiB, --moe-cache-lend with -b 2048
rem (prompt 126 tok/s at 2.2K tokens, 237 at 5.2K; -b 4096 with a 1400 MiB cache spills VRAM and halves it), decode 15-19 tok/s with Minecraft closed.
rem No MTP: its draft model needs ~1.5 GB of VRAM, which the 128K KV cache and the expert cache use (the draft then fails to load).
rem Thinking is off (--reasoning-budget 0). The model's chat template raises on the system message Claude Code sends after the user turn;
rem k448-chat-template.jinja (make-k448-template.py) renders it in place as a user turn instead, so no proxy is needed and the prompt cache keeps matching.
rem Close games first (Minecraft cut the speed by a third): the cache assumes ~8 GB of VRAM is free.
setlocal
set "M=C:\Projects\models\k448\Qwen3.8-Flash-Next-k448-IQ2_S.gguf"
if not defined LLAMA_BIN if exist "%~dp0build-clang\bin\llama-server.exe" set "LLAMA_BIN=%~dp0build-clang\bin"
if not defined LLAMA_BIN set "LLAMA_BIN=%~dp0build-win\bin"
if not exist "%~dp0slots" mkdir "%~dp0slots"
"%LLAMA_BIN%\llama-server.exe" -m "%M%" --alias k448 --host 127.0.0.1 --port 8090 --slot-save-path "%~dp0slots" ^
  -c 131072 -np 1 -t 10 -b 2048 -ub 512 --outputs-max 16 -ngl 999 -fit off --metrics ^
  -ctk q8_0 -ctv q8_0 -ot "per_layer_token_embd\.weight=CPU" -lzm on ^
  --cpu-moe --moe-cache-mib 950 --moe-cache-hybrid --moe-cache-adaptive --moe-cache-lend ^
  --jinja --chat-template-file "%~dp0k448-chat-template.jinja" --reasoning-budget 0 %*
