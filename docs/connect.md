# Connect your tools

`llama-server` speaks two APIs on the same port, so most chat apps, coding agents and SDKs connect with a base URL
and a key. extended-strata-moe changes how the model runs, not what the server speaks.

| API | endpoints | for |
|---|---|---|
| **OpenAI** | `/v1/chat/completions`, `/v1/responses`, `/v1/completions`, `/v1/models`, `/v1/embeddings` | the OpenAI SDKs, and every tool with an "OpenAI-compatible" setting |
| **Anthropic** | `/v1/messages`, `/v1/messages/count_tokens` | the Anthropic SDKs and Claude Code |

`tools/check_api.py` tests the server with the hybrid cache on, over both APIs, streamed and not, with and without
the API key. It passes.

- [Start the server](#start-the-server)
- [OpenAI-compatible tools](#openai-compatible-tools)
- [Claude Code and the Anthropic SDK](#claude-code-and-the-anthropic-sdk)
- [Good to know](#good-to-know)

## Start the server

```bash
.venv/bin/python tools/serve.py gpt-oss-120b-MXFP4.gguf --api-key my-secret-key
```

The expert cache needs no setup: it starts empty and fills itself with the experts your conversation uses
([how](reference.md#the-adaptive-cache)). A [profile](guide.md#1-trace-your-text) for your kind of text
(`--profile gpt-oss-120b.strp`) starts it warm, so the first replies are faster too.

It first measures what your GPU has to spare (loading the model once without the cache, see
[the guide](guide.md#3-run)), then prints the settings to give your tools, and starts `llama-server`:

```
GPU: 7910 of 8188 MiB free
  batch  4096: weights 1608 + KV 1305 + compute 3356 MiB, leaves 1241 MiB for the expert cache
  batch  2048: weights 1608 + KV 1233 + compute 1670 MiB, leaves 2999 MiB for the expert cache

Connect your tools to http://127.0.0.1:8080   (model name: gpt-oss-120b-MXFP4, API key: my-secret-key)

  OpenAI SDK / harness:    OPENAI_BASE_URL=http://127.0.0.1:8080/v1   OPENAI_API_KEY=my-secret-key
  Claude Code / Anthropic: ANTHROPIC_BASE_URL=http://127.0.0.1:8080   ANTHROPIC_AUTH_TOKEN=my-secret-key   ANTHROPIC_MODEL=gpt-oss-120b-MXFP4
  Browser chat:            http://127.0.0.1:8080
```

(That is a real run on the 8 GB laptop card: at the default 32k context a batch of 4096 would leave less than the 1,500 MiB
the expert cache needs, so it takes 2048.)

| option | default | |
|---|---|---|
| `--profile FILE` | none | the expert profile for your text ([`make_profile.py`](guide.md#1-trace-your-text)): the cache starts warm. Without it the cache starts empty and learns from the conversation |
| `--ubatch` | `auto` | the prompt batch size. Bigger is a faster prompt and a smaller expert cache; `auto` takes the biggest that leaves `--min-cache-mib` (1500) |
| `--favor` | `prompt` | `generation`: the biggest batch whose expert cache is within 3% of the largest. Faster answers, slower prompts (Qwen3-Coder-Next: 4.6 instead of 3.0 GB of cache, 33 instead of 31 tokens/s) |
| `--cache-mib N` | what is left | the expert cache's size, instead of what the measurement leaves |
| `--api-key KEY` | none | clients must send it, as `Authorization: Bearer KEY` (OpenAI) or `x-api-key: KEY` (Anthropic) |
| `--alias NAME` | the file name | the model name your tools send |
| `--host` | `127.0.0.1` | `127.0.0.1` is this computer only. `0.0.0.0` opens it to your network: set an API key first |
| `--port` | `8080` | |
| `--ctx` | `32768` | the context size. Coding agents need 32768 or more; a bigger context takes more VRAM, which the measurement takes into account |
| `--threads` | 5/8 of the logical CPUs | CPU threads. More than your fast cores can be much slower: [measure](guide.md#4-measure) |
| `--parallel` | `1` | slots (simultaneous requests); the hybrid is tuned for one |
| `-- ...` | | anything after `--` goes to `llama-server` as it is, e.g. `-- -ctk q8_0 -ctv q8_0` |

`--dry-run` prints the command without starting it.

## OpenAI-compatible tools

Any tool with a "base URL" setting works. Give it `http://127.0.0.1:8080/v1`, the key, and the model name. For tools
that read the standard variables:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=my-secret-key
```

The OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="my-secret-key")
reply = client.chat.completions.create(
    model="gpt-oss-120b-MXFP4",
    messages=[{"role": "user", "content": "Explain Mixture-of-Experts in two sentences."}],
)
print(reply.choices[0].message.content)
```

curl:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer my-secret-key" -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-120b-MXFP4","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Other tools: the base URL and key go in the provider settings (Open WebUI, Continue, Cline, Aider and similar all
have an "OpenAI compatible" provider). Streaming and the Responses API (`/v1/responses`) work as well.

## Claude Code and the Anthropic SDK

Claude Code talks to `/v1/messages`, which `llama-server` serves. Point it at your server with environment
variables:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_AUTH_TOKEN=my-secret-key
export ANTHROPIC_MODEL=gpt-oss-120b-MXFP4
claude
```

Notes:

- The base URL has **no `/v1`** at the end for Anthropic tools (they add the path), and **has `/v1`** for OpenAI
  tools.
- `ANTHROPIC_MODEL` should be the server's model name (`--alias`). If your tool also asks for a second, smaller model
  for side tasks and those requests fail, set that model name to the same value (see your tool's model settings).
- You are running a local open-weights model, not a Claude model: expect different quality, and check that your
  model supports tool calling, which agents depend on.

The Anthropic Python SDK:

```python
import anthropic

client = anthropic.Anthropic(base_url="http://127.0.0.1:8080", api_key="my-secret-key")
reply = client.messages.create(
    model="gpt-oss-120b-MXFP4",
    max_tokens=256,
    messages=[{"role": "user", "content": "Explain Mixture-of-Experts in two sentences."}],
)
print(reply.content[0].text)
```

`tools/check_api.py` runs the Anthropic endpoints through plain HTTP requests. It has not been run with the Claude
Code program itself, so tell us if something differs there.

## Good to know

- **Built for one user.** The hybrid is tuned and tested with one request at a time (`-np 1`, which `check_api.py`
  uses). Several slots (`-- -np 4`) work, but that is not benchmarked.
- **Long prompts.** The hybrid speeds up *generation*. A prompt is processed in big batches by llama.cpp's own path,
  which uploads the experts of the whole model over PCIe for every batch; so its speed is set by the batch size
  (`serve.py` takes the biggest that leaves room for the expert cache: on gpt-oss-120b 241 tokens/s at 2048, 397 at
  4096, 96 at llama.cpp's default 512). Coding agents send long prompts (tens of thousands of tokens);
  `llama-server` reuses the start of the previous prompt, so only the first request is slow. Watch the first request's
  time before judging.
- **Context size.** Set `--ctx` to what the tool needs. A bigger context costs VRAM, which comes out of the
  experts' share: plan with `moe_plan.py --reserve-gib` larger if loading runs out of memory.
- **Security.** The default host is `127.0.0.1`, so only programs on your computer reach the server. On a network,
  always set `--api-key`, and prefer a reverse proxy with TLS: the key travels in clear text over plain HTTP.
- **See what happened.** `--moe-cache-audit run.jsonl` (add it after `--`) writes one record per request: tokens,
  time, and how many experts came from the GPU. The server's `/metrics` and `tools/monitor.html` show it live.
