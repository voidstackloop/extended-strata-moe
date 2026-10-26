@echo off
rem usage: run.bat [threads]
set LLAMA=C:\Projects\llama.cpp-strata
cd /d %LLAMA%\build-clang\bin
C:\Projects\k448-engine\bench\bench.exe %1
