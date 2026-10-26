# extended-strata-moe 🚀

**Run massive Mixture-of-Experts (MoE) models on consumer GPUs at blistering speeds. 28+ tokens/s on 120B models with an 8 GB laptop GPU.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![llama.cpp PR](https://img.shields.io/badge/llama.cpp-PR%20%2330594-orange.svg)](https://github.com/ggml-org/llama.cpp/pull/30594)
[![MoE Architectures](https://img.shields.io/badge/MoE%20Architectures-53%20Verified-brightgreen.svg)](#supported-models)
[![Speedup](https://img.shields.io/badge/Speedup-Up%20to%20%2B109%25-red.svg)](#benchmarks)
[![API](https://img.shields.io/badge/API-OpenAI%20%26%20Anthropic-purple.svg)](#connect-your-tools)

> 💡 **What is this?**  
> An extension of [Strata](https://github.com/Niko1221/Strata) by [Niko1221](https://github.com/Niko1221). While original Strata was purpose-built for a single 125B model, **extended-strata-moe brings asynchronous GPU/CPU expert tiering to *every* MoE architecture supported by [llama.cpp](https://github.com/ggml-org/llama.cpp)** (DeepSeek V3/R1, Qwen 2.5/3 MoE, gpt-oss, GLM, Llama 4, Kimi, MiniMax, and 45+ others).

---

## ⚡ The Big Idea

Running a 60 GB – 120 GB MoE model usually requires a $10,000+ multi-GPU server. On a consumer GPU (8 GB – 16 GB VRAM), standard solutions hit severe bottlenecks:
* **Offloading whole layers to CPU (`--n-cpu-moe`)** leaves the GPU idle during CPU layers and crawls.
* **Uploading experts on-demand over PCIe (`--moe-cache-mib`)** chokes inference with continuous bus transfer stalls.

**extended-strata-moe solves this with true hybrid execution:**
1. **The Hot Experts live in VRAM**: The router experts your text actually uses are cached in GPU memory.
2. **CPU computes the misses in parallel**: While the GPU processes its cached experts, the CPU computes the remaining misses directly from system RAM. **Zero bytes uploaded over PCIe during generation.**
3. **The 3 µs CUDA Mailbox**: GPU and CPU synchronize in 3 µs via page-locked shared memory instead of 180 µs scheduler round trips.
4. **Adaptive Learning**: The cache automatically tracks which experts your conversation needs and swaps them in on the fly. **No manual tuning or offline profiling needed.**

```
                     ┌──────────────────────────────────────┐
                     │           Layer Input Token          │
                     └──────────────────┬───────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
       ┌────────────────────────┐              ┌────────────────────────┐
       │   GPU: Cached Experts  │              │  CPU: Uncached Experts │
       │     (resident in VRAM) │              │    (computed from RAM) │
       └────────────┬───────────┘              └────────────┬───────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │ (3 µs CUDA Mailbox sync)
                                        ▼
                     ┌──────────────────────────────────────┐
                     │          Sum & Next Layer            │
                     └──────────────────────────────────────┘
                          ✨ Zero PCIe Upload Stalls ✨
```

---

## 📊 Benchmarks: Real Performance on an 8 GB Laptop GPU

*Hardware: RTX 4060 Laptop (8 GB VRAM), i7-13620H, DDR5-5200, WSL2 Ubuntu.*

### 1. gpt-oss-120b (63 GB Model on 8 GB GPU)
| Configuration | Tokens / sec | % Reads from VRAM | Speedup vs Layer-Split |
|---|---:|---:|---:|
| llama.cpp `--cpu-moe` (Pure CPU experts) | 14.5 t/s | 0% | — |
| llama.cpp `--n-cpu-moe 33` (Best possible layer split) | 15.9 t/s | 8% | Baseline |
| **extended-strata-moe (Adaptive, Zero Config)** | **26.9 t/s** | **45%** | **+69% FASTER** 🚀 |
| **extended-strata-moe (Targeted Profile)** | **28.2 t/s** | **46%** | **+77% FASTER** 🚀 |

### 2. Qwen3-Coder-Next 80B (65.6 GB, Q6_K on 8 GB GPU)
| Configuration | Tokens / sec | Speedup |
|---|---:|---:|
| llama.cpp `--n-cpu-moe 46` (Best layer split) | 17.8 t/s | Baseline |
| **extended-strata-moe (Cold Start, 400 tokens)** | **30.9 t/s** | **+74%** 🚀 |
| **extended-strata-moe (Warmed Cache, 1,200 tokens)** | **37.2 t/s** | **+109% (2.1x FASTER)** 🚀 |

### 3. Tessera-125B-MoE "k448" (70.5 GiB, IQ2_S/IQ4_NL, on 8 GB GPU, native Windows)
| Configuration | Tokens / sec (short code answers) |
|---|---:|
| 128K context, 645 MiB expert cache (14% VRAM hits) | 17.5 t/s |
| 128K context, 3.2 GB expert cache (52% VRAM hits, 7.9 of 8.2 GB used) | 20 to 22.5 t/s |
| 64K context + MTP draft (`start-k448-fast.bat`) | 20 to 21.5 t/s (14.6 t/s on prose) |

One run per setting, no game running; a game cut it by a third. Setup, the Claude Code chat-template fix and the raw tables: [`docs/k448.md`](docs/k448.md).

> 🛡️ **Bit-Accurate Accuracy**: Verified against 53 MoE architectures in `test-llama-archs`. KL divergence versus full CPU evaluation is **< 0.002** (identical to llama.cpp's own internal GPU vs CPU rounding tolerance).

---

## 🎯 Is This For You?

| Your Setup | Should You Use extended-strata-moe? |
|---|---|
| **MoE model larger than your VRAM** (e.g. 20B–120B+ on 8 GB, 12 GB, or 16 GB GPU) + enough system RAM | **YES — Huge performance improvement (up to 2x+ faster)** |
| **Dense model** (Llama 3, Gemma 2, Mistral 7B) | No — Dense models have no experts to route; run standard llama.cpp |
| **Model fits entirely in VRAM** | No — If it fits 100% in VRAM, full GPU offload is already fastest |
| **Uniform routing models** (e.g. Mixtral 8x7B) | Marginal — Works best when models have skewed routing distributions |

---

## 🚀 Quick Start (60 Seconds)

### 1. Build
Clone and run the automated setup script (builds patched llama.cpp and tools):

```bash
git clone https://github.com/voidstackloop/extended-strata-moe.git
cd extended-strata-moe
bash scripts/setup.sh
```

### 2. Serve Any MoE Model
Launch the self-tuning server. It measures your available VRAM, sizes the cache, enables adaptive hybrid execution, and hosts an API:

```bash
.venv/bin/python tools/serve.py /path/to/model-MoE.gguf --api-key my-secret-key
```

*That's it! The server automatically learns which experts your conversation needs and dynamically swaps them into VRAM.*

---

## 🔌 Connect Your Tools (OpenAI & Anthropic Compatible)

`tools/serve.py` serves both OpenAI and Anthropic endpoints simultaneously on a single port:

```
http://127.0.0.1:8080
```

### Use with Claude Code
```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"
export ANTHROPIC_AUTH_TOKEN="my-secret-key"
export ANTHROPIC_MODEL="gpt-oss-120b"
claude
```

### Use with Cursor / LibreChat / OpenCode / OpenAI SDK
```bash
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="my-secret-key"
```

### Python
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="my-secret-key")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain asynchronous expert offloading in simple terms."}]
)
print(response.choices[0].message.content)
```

---

## 🧰 Built-in Developer Tools

| Tool | What it does |
|---|---|
| `tools/serve.py` | One-command server: auto-allocates VRAM, enables adaptive cache, hosts OpenAI & Anthropic API. |
| `tools/make_profile.py` | Create a `.strp` profile from sample text to start the cache 100% warm from token zero. |
| `tools/monitor.html` | Real-time browser dashboard showing live token rates, VRAM hit rate, PCIe bandwidth, and per-layer stats. |
| `tools/bench_cache.py` | Interleaved benchmarking tool to measure performance against baseline `--n-cpu-moe`. |
| `tools/check_api.py` | Automated test suite verifying OpenAI/Anthropic endpoints, streaming, and auth. |
| `tools/check_audit.py` | Validates JSONL audit trail (`--moe-cache-audit`) for hit rates and hardware timings. |

### Watch It Live in Your Browser
Start the monitor UI to visualize expert hits and memory efficiency in real time:
```bash
python3 -m http.server -d tools 8090
# Open in your browser:
# http://127.0.0.1:8090/monitor.html?server=http://127.0.0.1:8080
```

---

## 🧠 Supported Architectures

Tested and verified against all 53 MoE architectures in `llama.cpp`:

* **DeepSeek**: DeepSeek V2, DeepSeek V3, DeepSeek R1, DeepSeek V4
* **Qwen**: Qwen1.5 MoE, Qwen2 MoE, Qwen3-30B-A3B / 235B-A22B, Qwen3-Next, Qwen3.5 MoE, Qwen3-VL MoE
* **GPT-OSS**: gpt-oss-20b, gpt-oss-120b
* **GLM**: GLM-4.5 / 4.6, GLM-5.2, GLM-5.3-Flash
* **Llama**: Llama 4 Scout / Maverick, Mixtral
* **Others**: Kimi K2/K3, MiniMax M2/M3/01, Granite MoE, OLMoE, Phi-3.5-MoE, DBRX, Snowflake Arctic, Grok, and more.

---

## 🤝 Upstream Status

A pull request proposing this hybrid expert cache is active on official `llama.cpp`:
* [**llama.cpp PR #30594: hybrid MoE expert cache with profile prefilling and zero-upload inference**](https://github.com/ggml-org/llama.cpp/pull/30594)

---

## 📚 Deep Dive & Documentation

* [docs/guide.md](docs/guide.md): Comprehensive walkthrough, VRAM sizing formulas, thread tuning.
* [docs/connect.md](docs/connect.md): Detailed API integrations for IDEs, agents, and apps.
* [docs/reference.md](docs/reference.md): CLI flags, C API, Prometheus metrics, and audit schema.
* [docs/ADR-001](docs/ADR-001-generic-moe-expert-tiering.md): Architecture Decision Record — Generic MoE Tiering.
* [docs/ADR-002](docs/ADR-002-cpu-misses-not-uploads.md): Architecture Decision Record — CPU Misses vs PCIe Uploads.

---

## 📜 Credits & Acknowledgements

* **[Strata](https://github.com/Niko1221/Strata) by [Niko1221](https://github.com/Niko1221)**: The breakthrough expert-tiering concept, `STRP` profile format, and routing trace design.
* **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: The core C++ inference engine and community.

Licensed under [MIT](LICENSE).
