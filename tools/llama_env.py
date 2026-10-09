"""tools/llama_env.py - where the strata-moe build of llama.cpp is, and the environment to run its binaries in.

The binaries: $LLAMA_BIN, else llama.cpp/build/bin in this repo (what scripts/setup.sh builds), else
~/projects/llama.cpp/build-cuda/bin (tools/build_cuda.sh).
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def default_bin():
    if os.environ.get("LLAMA_BIN"):
        return os.environ["LLAMA_BIN"]
    local = REPO / "llama.cpp" / "build" / "bin"
    return str(local if local.is_dir() else Path.home() / "projects/llama.cpp/build-cuda/bin")


def env(bin_dir):
    """os.environ with the build's shared libraries first, then a user-local CUDA and WSL's driver if present."""
    first = [bin_dir] + [p for p in (str(Path.home() / "cuda-13.4/lib"), "/usr/lib/wsl/lib") if Path(p).is_dir()]
    rest = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    return dict(os.environ, LD_LIBRARY_PATH=":".join(first + rest))
