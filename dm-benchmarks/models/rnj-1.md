# Essential AI Rnj-1 8B (`rnj-1`)

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `rnj-1:8b` / `rnj-1:latest` / `rnj-1:8b-instruct-q4_K_M` (`d20e29ab8d0f`), `rnj-1:8b-instruct-q8_0` (`57eea2304a52`) *(plus `8b-instruct-fp16` 17 GB — Too Large)* |
| **Parameters (Total / Active)** | `8.3B` / `8.3B` (dense) |
| **Architecture** | Dense global-attention transformer (`32` layers, `32` attn heads, `8` KV heads, `head_dim=128`, GeGLU, tied embeddings) |
| **Context Window (Max / Practical M4)** | `32K` (`128K` via YaRN) / `8k–16k` practical M4 *(global attention across all 32 layers)* |
| **Recommended Quantization** | `rnj-1:8b-instruct-q4_K_M` (`5.1 GB`, `8b`, `latest`) or `8b-instruct-q8_0` (`8.8 GB`) |
| **License** | **Apache-2.0** |
| **pydantic-ai Readiness** | `Out-of-the-box` (`OllamaModel` `/v1`; uses Hermes `<tool_call>` XML format, no `<think>` overhead) |
| **Status** | `Candidate` (updated **11 months ago**; outperformed by 2026 `<10B` peers on reasoning/agents, but clean non-thinking Apache-2.0 8B STEM/code tool model) |
| **Last Updated** | `2026-11-26` |

## Ollama Tags & Mac Mini M4 (24GB) Fit

| Ollama Tag | Hash | Disk Size | Backend | Ctx | M4 24GB Fit |
| --- | --- | ---: | --- | ---: | --- |
| `rnj-1:8b` / `latest` / `8b-instruct-q4_K_M` | `d20e29ab8d0f` | **5.1 GB** | GGUF | 32K | `Comfortable` |
| `rnj-1:8b-instruct-q8_0` | `57eea2304a52` | 8.8 GB | GGUF | 32K | `Comfortable` |
| `rnj-1:8b-instruct-fp16` | `cf0061bce5f2` | 17 GB | GGUF | 32K | `Too Large` |

- **Weight Size (Disk / Loaded)** `[Primary]`: `5.1 GB` (`q4_K_M`), `8.8 GB` (`q8_0`).
- **KV Cache Math** `[Estimated]`: All 32 layers use global attention (`8` KV heads × `128` `head_dim` → **128 KB/token** fp16: ~1.0 GB @ 8k, ~2.0 GB @ 16k, ~4.0 GB @ 32k; use `OLLAMA_KV_CACHE_TYPE=q8_0` for ≥16k).
- **Est. Total RAM @ 8k / 16k / 32k** `[Estimated]`: **~6.8–9.8 GB** (`q4_K_M`, `Comfortable`), **~10.5–13.5 GB** (`q8_0`, `Comfortable`–`Tight`).
- **Fit Tier**: `Comfortable` (`q4_K_M`, `q8_0` ≤16k); `Too Large` (`fp16`).
- **Est. M4 Decode Speed** `[Estimated]` (`~120 GB/s ÷ size_GB × 0.65–0.80`):
  - `8b-instruct-q4_K_M` (5.1 GB): **~15.3–18.8 tok/s**
  - `8b-instruct-q8_0` (8.8 GB): **~8.9–10.9 tok/s**

## Decision-Model Assessment

- **Decision Quality & Determinism** `[Primary]`: Trained from scratch (~8.4T tokens) for code, math, and STEM (`GSM8k 92.6`, `Minerva-MATH-500 88.4`, `MMLU-STEM 81.8`, `HE+ 83.5`). However, it is **11 months old** and trails 2026 `<10B` peers on hard reasoning (`GPQA Diamond 38.9` vs. `Ornith-1.5-9B 86.4`, `Qwen3.5-9B 80.4+`, `Granite-4.2-8B 64.14`) and `SWE-bench` (`20.8%` bash-only vs. `Ornith-1.5-9B 70.6%` OpenHands).
- **Structured Output & JSON Schema** `[Primary]`: GGUF tags support `llama.cpp` grammar-constrained `json_schema` decoding for `pydantic-ai` `NativeOutput`.
- **Tool & Function Calling** `[Primary]`: Native `tools` capability on Ollama using Hermes `<tool_call>...</tool_call>` syntax (`BFCL` highlight on Ollama/HF; exact `BFCL` score not numerically printed in the HF table).
- **Thinking / Latency Tradeoff** `[Primary]`: **No `<think>` reasoning block** (`['tools']` only), yielding predictable single-turn latency at ~15–19 tok/s (`q4_K_M`) without CoT token bloat.
- **Operational & pydantic-ai Readiness** `[Primary]`: **`Out-of-the-box`** via `OllamaModel` (`'ollama:rnj-1:8b-instruct-q4_K_M'`, `/v1`). Supports `NativeOutput` and `ToolOutput` cleanly via Hermes tool parsing.

## Benchmarks & Evidence

Essential AI reported benchmarks (`EssentialAI/rnj-1-instruct` README, `[Primary]`):

- **Coding**: `HE+` `83.5` | `MBPP+` `75.7` | `BigCodeBench-full` `57.1` | `SWE-bench (bash)` `20.8` | `Enamel` `49.0`
- **Math & Science**: `GSM8k` `92.6` | `Minerva-MATH-500` `88.4` | `AIME 25` `43.3` (`53.3` @ 128k YaRN) | `MMLU-STEM` `81.8` (`69.4` @ 128k YaRN) | `GPQA-Diamond` `38.9` (`41.4` @ 128k YaRN)

## Caveats & Open Questions

- **Older Release (11 Months Ago)**: Significantly outperformed by newer 2026 models (`ornith-1.5:9b`, `qwen3.5:9b`, `granite4.2:8b`, `mimo-v2.6:9b`) on general reasoning, agentic tasks, and `GPQA`.
- **Text-Only & 32K Context**: No vision support; `128K` YaRN extrapolation degrades `MMLU-STEM` (`81.8` → `69.4`).
- **No Local M4 Measurements Yet**: Already pulled locally (`rnj-1:8b` in your `ollama list`), so `NativeOutput`/`ToolOutput` M4 tests can be run anytime.

## Sources

- `[2026-11-26]` [Ollama Library: rnj-1](https://ollama.com/library/rnj-1) & [rnj-1/tags](https://ollama.com/library/rnj-1/tags) — `8b` GGUF tags (`5.1–17 GB`), `32K` ctx, `tools` capability, updated 11 months ago.
- `[2026-11-26]` [Hugging Face: EssentialAI/rnj-1-instruct](https://huggingface.co/EssentialAI/rnj-1-instruct/raw/main/README.md) — Apache-2.0 license, `8.3B` dense architecture, Hermes `<tool_call>` parser, benchmark table.
