@echo off
rem k448 "fast code" profile: 64K context + MTP draft head (2 draft tokens), port 8091. Same model and chat template as start-k448.bat.
rem Measured 2026-10-27 (RTX 4060 Laptop, Minecraft closed): short code answers 20-21.5 tok/s (draft acceptance 93%%) against 17-19 for
rem start-k448.bat; prose and code explanations 14.6 tok/s (acceptance 65%%), the same as without MTP; prompt 170 tok/s at 2.2K tokens (-ub 256).
rem The MTP draft model needs ~1.5 GB of VRAM, so the context is 64K (KV q8_0 918 MiB) and the expert cache only 600 MiB. Use start-k448.bat
rem (128K, no MTP) for long agent sessions. Only one of the two can run at a time (one GPU).
setlocal
set "M=C:\Projects\models\k448\Qwen3.8-Flash-Next-k448-IQ2_S.gguf"
set "D=C:\Projects\models\k448\mtp-Qwen3.8-Flash-Next-balanced.gguf"
if not defined LLAMA_BIN if exist "%~dp0build-clang\bin\llama-server.exe" set "LLAMA_BIN=%~dp0build-clang\bin"
if not defined LLAMA_BIN set "LLAMA_BIN=%~dp0build-win\bin"
if not exist "%~dp0slots" mkdir "%~dp0slots"
"%LLAMA_BIN%\llama-server.exe" -m "%M%" --alias k448 --host 127.0.0.1 --port 8091 --slot-save-path "%~dp0slots" ^
  -c 65536 -np 1 -t 10 -b 2048 -ub 256 --outputs-max 16 -ngl 999 -fit off --metrics ^
  -ctk q8_0 -ctv q8_0 -ot "per_layer_token_embd\.weight=CPU" -lzm on ^
  --cpu-moe --moe-cache-mib 600 --moe-cache-hybrid --moe-cache-adaptive --moe-cache-lend ^
  --jinja --chat-template-file "%~dp0k448-chat-template.jinja" --reasoning-budget 0 ^
  --spec-type draft-mtp -md "%D%" --spec-draft-n-max 2 %*
