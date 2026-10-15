"""tools/serve.py - start llama-server with the hybrid cache sized to your GPU, and print how to connect your tools to it.

    python tools/serve.py MODEL.gguf [--profile model.strp] [--api-key KEY] [--port 8080] [--alias NAME]
        [--ctx 32768] [--threads 6] [--host 127.0.0.1] [--ubatch auto|N] [-- extra llama-server flags]

What it fits to the card, by measuring (a dry run loads the model without the cache and reads llama.cpp's own buffer
sizes): the weights, the KV cache for --ctx and the compute buffer for the batch size, and gives what is left to the
expert cache.  The batch size matters twice: every prompt batch uploads the experts of the whole model over PCIe, so
a bigger batch means a faster prompt (gpt-oss-120b: 95 tokens/s at 512, 400 at 4096), but its compute buffer takes
VRAM from the cache.  --ubatch auto takes the biggest batch that leaves --min-cache-mib for the cache.

The server speaks the OpenAI API (/v1/chat/completions, /v1/responses, /v1/completions, /v1/models) and the
Anthropic API (/v1/messages, /v1/messages/count_tokens) on the one port.  The expert cache is adaptive
(--moe-cache-adaptive): it swaps the experts your text uses most into VRAM while it generates, so it needs no
profile; --profile only starts it warm.  --bin DIR: see tools/llama_env.py.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llama_env

BUFFER = re.compile(r"\b(?:CUDA|ROCm|HIP|Vulkan)\d+\s+(model|KV|compute|RS) buffer size =\s+([0-9.]+) MiB")


def gpu_mib():
    """(total, free) MiB of the first GPU, from nvidia-smi; None where there is none to ask"""
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    try:
        out = subprocess.check_output([smi, "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
                                      text=True, timeout=30).splitlines()[0]
        total, free = (int(x) for x in out.split(","))
        return total, free
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


# the flags that change what the GPU holds, with how many values each takes: the dry run gets these and nothing else
MEMORY_FLAGS = {"-ctk": 1, "--cache-type-k": 1, "-ctv": 1, "--cache-type-v": 1, "-fa": 1, "--flash-attn": 1, "-kvu": 0,
                "--kv-unified": 0, "--swa-full": 0, "-nkvo": 0, "--no-kv-offload": 0, "-ts": 1, "--tensor-split": 1,
                "-sm": 1, "--split-mode": 1, "-mg": 1, "--main-gpu": 1,
                "--load-mode": 1}   # not a memory flag, but the dry run must load the model the way the server will (a slow disk)


def memory_flags(extra):
    out, i = [], 0
    while i < len(extra):
        n = MEMORY_FLAGS.get(extra[i])
        if n is not None:
            out += extra[i:i + 1 + n]
            i += 1 + n
        else:
            i += 1
    return out


def measure(bin, model, ctx, ub, extra, env):
    """MiB the GPU holds without a cache: {'model', 'KV', 'compute', 'RS'}, from llama.cpp's own log"""
    cmd = [str(Path(bin) / "llama-completion"), "-m", model, "-c", str(ctx), "-b", str(ub), "-ub", str(ub), "-ngl", "999",
           "-fit", "off", "--cpu-moe", "-n", "1", "-p", "Hi", "-no-cnv", "--no-warmup", "-np", "1", "-v"] + memory_flags(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL, timeout=900)
    sums = {"model": 0.0, "KV": 0.0, "compute": 0.0, "RS": 0.0}
    for kind, mib in BUFFER.findall(r.stdout + r.stderr):
        sums[kind] += float(mib)
    if sums["model"] == 0 or sums["compute"] == 0:
        sys.exit("could not read the GPU buffer sizes from llama.cpp's log (is this a GPU build?):\n" + (r.stderr or r.stdout)[-600:])
    return sums


def main():
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("model")
    ap.add_argument("--profile", help="expert profile (.strp) from tools/make_profile.py: the cache starts warm "
                                      "(without it, it starts empty and learns from the conversation)")
    ap.add_argument("--cache-mib", type=int, help="VRAM for experts, instead of what is left after the buffers")
    ap.add_argument("--ubatch", default="auto", help="prompt batch size: auto, or 512 .. 8192 (default: auto)")
    ap.add_argument("--min-cache-mib", type=int, default=1500, help="auto: the batch size must leave this for the cache")
    ap.add_argument("--favor", choices=["prompt", "generation"], default="prompt",
                    help="auto: prompt = the biggest batch that leaves --min-cache-mib (long prompts: coding agents); generation = "
                         "the biggest batch whose cache is within 3%% of the largest (Qwen3-Coder-Next: 4.6 instead of 3.0 GB, "
                         "+5%% tokens/s, prompts slower)")
    ap.add_argument("--margin-mib", type=int, default=400, help="VRAM left free (CUDA context, other programs)")
    ap.add_argument("--alias", help="the model name clients send (default: the file name)")
    ap.add_argument("--api-key", help="clients must send it (Authorization: Bearer KEY, or x-api-key: KEY)")
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only; 0.0.0.0 = the network")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx", type=int, default=32768, help="context size; coding agents want 32768 or more")
    ap.add_argument("--parallel", type=int, default=1, help="slots (simultaneous requests); the hybrid is tuned for 1")
    ap.add_argument("--threads", type=int, default=max(4, round(0.625*(os.cpu_count() or 8))),
                    help="CPU threads (default: 5/8 of the logical CPUs, 10 on a 6P+4E laptop; more than the fast cores "
                         "can be much slower: sweep with tools/bench_cache.py -t)")
    ap.add_argument("--bin", default=llama_env.default_bin())
    ap.add_argument("--dry-run", action="store_true", help="print the command, do not start it")
    a, extra = ap.parse_known_args()
    extra = [x for x in extra if x != "--"]
    env = llama_env.env(a.bin)

    if a.host != "127.0.0.1" and not a.api_key:
        print("warning: listening on the network without --api-key: anyone who can reach this port can use the model",
              file=sys.stderr)

    cache_mib, ub = a.cache_mib or 0, None
    gpu = gpu_mib()
    if a.profile and gpu is None and not a.cache_mib:
        sys.exit("no GPU found to size the cache for: give --cache-mib")
    if a.ubatch != "auto":
        candidates = [int(a.ubatch)]
    else:
        candidates = [4096, 2048, 1024, 512]
    if gpu is not None:
        total, free = gpu
        print(f"GPU: {free} of {total} MiB free", file=sys.stderr)
        fits = {}
        for ub in candidates:
            m = measure(a.bin, a.model, a.ctx, ub, extra, env)
            fixed = m["model"] + m["KV"] + m["RS"] + m["compute"]
            left = int(free - fixed - a.margin_mib)
            fits[ub] = left
            print(f"  batch {ub:>5}: weights {m['model']:.0f} + KV {m['KV']:.0f} + compute {m['compute']:.0f} MiB"
                  f"{' + RS %.0f' % m['RS'] if m['RS'] else ''}, leaves {max(left, 0)} MiB for the expert cache", file=sys.stderr)
            if a.favor == "prompt" and (left >= a.min_cache_mib or ub == candidates[-1]):
                break
        if a.favor == "generation":   # every size was measured: the biggest batch that costs the cache at most 3%
            best = max(fits.values())
            ub = max(u for u, l in fits.items() if l >= 0.97*best)
            left = fits[ub]
            print(f"  --favor generation: batch {ub}", file=sys.stderr)
        if not a.cache_mib:
            cache_mib = max(left, 0)
        if left < 0 and not a.cache_mib:
            print("warning: the weights, KV cache and compute buffer alone do not fit the free VRAM: lower --ctx or "
                  "--ubatch", file=sys.stderr)

    alias = a.alias or Path(a.model).stem
    cmd = [str(Path(a.bin) / "llama-server"), "-m", a.model, "-ngl", "999", "-fit", "off", "-t", str(a.threads),
           "-c", str(a.ctx), "-np", str(a.parallel), "--alias", alias, "--host", a.host, "--port", str(a.port), "--metrics"]
    if ub:
        cmd += ["-b", str(ub), "-ub", str(ub)]
    cmd += ["--cpu-moe"]
    if cache_mib >= 256:   # the adaptive cache follows the conversation; a profile only starts it warm
        cmd += ["--moe-cache-mib", str(cache_mib), "--moe-cache-hybrid", "--moe-cache-adaptive"]
        if a.profile:
            cmd += ["--moe-cache-profile", a.profile]
        else:
            print("no --profile: the expert cache starts empty and fills itself with the experts your text uses. "
                  "A profile for your text starts it warm:\n"
                  f"    python tools/make_profile.py {a.model} my_text.txt", file=sys.stderr)
    elif a.profile:
        print("note: too little VRAM left for a useful expert cache; running without it", file=sys.stderr)
    if a.api_key:
        cmd += ["--api-key", a.api_key]
    cmd += extra

    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '::') else a.host}:{a.port}"
    key = a.api_key or "none"
    print(f"""
Connect your tools to {url}   (model name: {alias}, API key: {key})

  OpenAI SDK / harness:    OPENAI_BASE_URL={url}/v1   OPENAI_API_KEY={key}
  Claude Code / Anthropic: ANTHROPIC_BASE_URL={url}   ANTHROPIC_AUTH_TOKEN={key}   ANTHROPIC_MODEL={alias}
  Browser chat:            {url}
""", flush=True)
    if a.dry_run:
        print(" ".join(cmd))
        return 0
    os.execve(cmd[0], cmd, env)


if __name__ == "__main__":
    sys.exit(main())
