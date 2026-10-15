"""tools/check_api.py - the OpenAI and Anthropic APIs on a llama-server running the hybrid cache.

Starts the server through tools/serve.py with an API key and the audit log, then checks:
  OpenAI:     /v1/models, /v1/chat/completions (plain and streamed), /v1/responses
  Anthropic:  /v1/messages (plain and streamed), /v1/messages/count_tokens
  auth:       no key and a wrong key are refused (401); "Authorization: Bearer" and "x-api-key" both work
  the cache:  every generation left a request record in the audit log, and the hybrid uploaded nothing

    python tools/check_api.py MODEL.gguf PROFILE.strp [--cache-mib 256] [--bin DIR]
"""
import argparse
import json
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llama_env

PORT, KEY, ALIAS = 8098, "test-key", "strata-test"
BASE = f"http://127.0.0.1:{PORT}"


def call(path, body=None, headers=None, stream=False):
    h = {"Content-Type": "application/json", **(headers if headers is not None else {"Authorization": f"Bearer {KEY}"})}
    req = urllib.request.Request(BASE + path, method="POST" if body is not None else "GET", headers=h,
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read().decode()
    return raw if stream else json.loads(raw)


def sse(raw):
    """[(event, data)] of a server-sent-events body; data is the parsed JSON, or the string [DONE]"""
    out = []
    for block in raw.strip().split("\n\n"):
        ev, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if data is not None:
            out.append((ev, data if data == "[DONE]" else json.loads(data)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("profile")
    ap.add_argument("--cache-mib", type=int, default=256)
    ap.add_argument("--bin", default=llama_env.default_bin())
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as d:
        audit = Path(d) / "audit.jsonl"
        cmd = [sys.executable, "-I", str(Path(__file__).parent / "serve.py"), a.model, "--profile", a.profile, "--cache-mib",
               str(a.cache_mib), "--alias", ALIAS, "--api-key", KEY, "--port", str(PORT), "--ctx", "4096", "--bin", a.bin,
               "--", "-np", "1", "--no-warmup", "--moe-cache-audit", str(audit)]
        srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(240):
                try:
                    urllib.request.urlopen(BASE + "/health", timeout=1)
                    break
                except OSError:
                    time.sleep(0.5)
            else:
                raise AssertionError("the server did not come up")

            # auth
            for headers in ({}, {"Authorization": "Bearer wrong"}, {"x-api-key": "wrong"}):
                try:
                    call("/v1/models", headers=headers)
                    raise AssertionError(f"{headers or 'no key'} was accepted")
                except urllib.error.HTTPError as e:
                    assert e.code == 401, e.code
            models = call("/v1/models", headers={"x-api-key": KEY})
            assert any(m["id"] == ALIAS for m in models["data"]), models
            print(f"auth: ok - no key and wrong keys refused; Bearer and x-api-key accepted; /v1/models lists {ALIAS}")

            # OpenAI
            msgs = [{"role": "user", "content": "Say hello in one short sentence."}]
            r = call("/v1/chat/completions", {"model": ALIAS, "messages": msgs, "max_tokens": 16, "temperature": 0})
            assert r["choices"][0]["message"]["content"].strip() and r["usage"]["completion_tokens"] > 0, r
            raw = call("/v1/chat/completions", {"model": ALIAS, "messages": msgs, "max_tokens": 16, "temperature": 0, "stream": True},
                       stream=True)
            ev = sse(raw)
            text = "".join(ch["delta"].get("content") or "" for _, d in ev[:-1] for ch in d["choices"])
            assert ev[-1][1] == "[DONE]" and text.strip(), raw[-300:]
            r = call("/v1/responses", {"model": ALIAS, "input": "Say hello in one short sentence.", "max_output_tokens": 16,
                                       "temperature": 0})
            assert r["status"] == "completed" and r["output"], r
            print("openai: ok - chat/completions (plain, streamed), responses")

            # Anthropic
            body = {"model": ALIAS, "max_tokens": 16, "temperature": 0, "system": "Be brief.", "messages": msgs}
            r = call("/v1/messages", body, headers={"x-api-key": KEY, "anthropic-version": "2023-06-01"})
            assert r["type"] == "message" and r["role"] == "assistant" and r["content"][0]["type"] == "text", r
            assert r["content"][0]["text"].strip() and r["usage"]["output_tokens"] > 0 and r["stop_reason"], r
            raw = call("/v1/messages", {**body, "stream": True}, headers={"x-api-key": KEY}, stream=True)
            names = [e for e, _ in sse(raw)]
            assert names[0] == "message_start" and names[-1] == "message_stop" and "content_block_delta" in names, names
            n = call("/v1/messages/count_tokens", {k: body[k] for k in ("model", "system", "messages")}, headers={"x-api-key": KEY})
            assert n["input_tokens"] > 0, n
            print(f"anthropic: ok - messages (plain, streamed), count_tokens ({n['input_tokens']} input tokens)")
        finally:
            srv.send_signal(signal.SIGINT)
            try:
                srv.wait(timeout=60)
            except subprocess.TimeoutExpired:
                srv.kill()
        recs = [json.loads(line) for line in audit.read_text().splitlines()]

    reqs = [r for r in recs if r["event"] == "request"]
    assert len(reqs) == 5, [r["event"] for r in recs]   # chat x2, responses, messages x2
    assert all(r["n_gen"] > 0 and r["req_hits"] + r["req_misses"] > 0 and r["req_bytes_upload"] == 0 for r in reqs), reqs
    ex = [r for r in recs if r["event"] == "exit"][0]
    print(f"cache: ok - {len(reqs)} request records, {sum(r['req_hits'] for r in reqs)} hits and "
          f"{sum(r['req_misses'] for r in reqs)} misses, {ex['bytes_upload']} bytes uploaded, {ex['perf']['gen_tps']:.1f} t/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
