@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo vcvars failed & exit /b 1)
set LLAMA=C:\Projects\llama.cpp-strata
cd /d C:\Projects\k448-engine\bench
"C:\Projects\llvm\clang+llvm-23.1.2-x86_64-pc-windows-msvc\bin\clang-cl.exe" /nologo /O2 /I %LLAMA%\ggml\include bench.c /Fe:bench.exe /link /LIBPATH:%LLAMA%\build-clang\ggml\src ggml.lib ggml-base.lib
if errorlevel 1 (echo compile failed & exit /b 1)
echo built
