"""tools/make_profile.py - the expert profile for YOUR text, in one command.

    python tools/make_profile.py MODEL.gguf my_text.txt [-o model.strp] [--threads 8] [--ctx 4096]

It runs the model on the text on the CPU (the routing tracer), ranks every expert by how much that text uses it, and writes
the profile that `serve.py --profile` and `--moe-cache-profile` read.  The profile does not depend on the cache size.

Use text like what you will run: your chats, your code, documents in your language.  A few thousand tokens is enough
(15,000 tokens of gpt-oss-120b take about 18 minutes on 10 threads); more gives a steadier ranking.  A profile made from
other text still works, with fewer hits: on gpt-oss-120b, 24% of the experts served from VRAM with a profile from
llama.cpp's docs against 44% with one from the same kind of text as the prompt.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="the GGUF (the first shard of a split model)")
    ap.add_argument("text", help="a text file in the style you will run")
    ap.add_argument("-o", "--out", help="the profile to write (default: the model's name + .strp)")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads for the tracer")
    ap.add_argument("--ctx", type=int, default=4096, help="the tracer's context size")
    ap.add_argument("--tracer", default=str(ROOT / "build" / "routing_trace"), help="the tracer, built by scripts/setup.sh")
    a = ap.parse_args()

    if not Path(a.tracer).exists():
        sys.exit(f"{a.tracer} not found: run scripts/setup.sh first (it builds the tracer)")
    out = Path(a.out) if a.out else Path(a.model).with_suffix(".strp").name
    with tempfile.TemporaryDirectory() as d:
        trace = Path(d) / "trace.bin"
        print(f"tracing {a.text} on the CPU ...", flush=True)
        subprocess.run([a.tracer, a.model, a.text, str(trace), str(a.ctx), str(a.threads)], check=True)
        print("ranking the experts ...", flush=True)
        # any size works: the profile is a ranking, and llama.cpp fills as many as the cache has slots for
        subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "moe_plan.py"), a.model, "--cache-mib", "4096",
                        "--trace", str(trace), "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
    print(f"wrote {out}\nrun it with:  python tools/serve.py {a.model} --profile {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
