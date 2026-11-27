# IBM Granite 4.2 (`granite4.2`)

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `granite4.2:8b` / `8b-q4_K_M` / `latest` (`f586c02fdecd`), `granite4.2:8b-q8_0` (`62837636dd31`), `granite4.2:3b` / `3b-q4_K_M` (`40577dc168a3`), `granite4.2:3b-q8_0` (`f8163a0ed616`) *(plus `8b-bf16` & `30b` — Too Large)* |
| **Parameters (Total / Active)** | `8B` / `8B` · `3B` / `3B` (dense) |
| **Architecture** | `GraniteForCausalLM` (dense decoder-only transformer: 40 layers, 8 KV heads, `head_dim=128` on 8B / `64` on 3B, SwiGLU, RoPE) |
| **Context Window (Max / Practical M4)** | `128K` native (`512K` extended) / `8k–32k` practical M4 *(use `OLLAMA_KV_CACHE_TYPE=q8_0` above 16k on 8B)* |
| **Recommended Quantization** | `granite4.2:8b-q4_K_M` (`5.3 GB`, `latest`) or `8b-q8_0` (`9.3 GB`); `granite4.2:3b-q8_0` (`3.9 GB`) for lightweight routing |
| **License** | **Apache-2.0** |
| **pydantic-ai Readiness** | `Minor config` (`OllamaModel` `/v1`; **caveat**: Ollama registry shows `[]` capability badges—verify `tools` support & override Modelfile `temp=1.0` default) |
| **Status** | `Shortlisted` (top Apache-2.0 enterprise instruction-following & tunable-effort thinking `<10B`) |
| **Last Updated** | `2026-11-26` |

## Ollama Tags & Mac Mini M4 (24GB) Fit

All `<10B` tags on Ollama are GGUF (`Text`-only, `128K` ctx, released Aug 25, 2026):

| Ollama Tag | Hash | Disk Size | M4 24GB Fit | Notes |
| --- | --- | ---: | --- | --- |
| `granite4.2:3b` / `3b-q4_K_M` | `40577dc168a3` | 2.2 GB | `Comfortable` | ~3.0–3.5 GB est. RAM @ 8k–16k |
| `granite4.2:3b-q8_0` | `f8163a0ed616` | 3.9 GB | `Comfortable` | ~4.8–5.3 GB est. RAM @ 8k–16k |
| `granite4.2:8b` / `8b-q4_K_M` / `latest` | `f586c02fdecd` | **5.3 GB** | `Comfortable` | ~7.0–8.5 GB est. RAM @ 8k–16k |
| `granite4.2:8b-q8_0` | `62837636dd31` | 9.3 GB | `Comfortable`–`Tight` | ~11.2–13.5 GB est. RAM @ 8k–32k |
| `granite4.2:8b-bf16` | `6ec214bd57a2` | 18 GB | `Too Large` | Exceeds ~16–18 GB usable M4 ceiling |
| `granite4.2:30b` | `be22829aa92a` | 18 GB | `Too Large` | 30B dense; exceeds M4 budget & >10B |

- **Weight Size (Disk / Loaded)** `[Primary]`: `5.3 GB` (`8b-q4_K_M`), `9.3 GB` (`8b-q8_0`), `2.2–3.9 GB` (`3b`).
- **KV Cache Math** `[Estimated]`: Dense GQA with 40 layers × 8 KV heads (`head_dim=128` on 8B → **160 KB/token** in fp16: ~1.25 GB @ 8k, ~2.5 GB @ 16k, ~5.0 GB @ 32k). Enable `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` to halve KV overhead for ≥16k contexts.
- **Fit Tier**: `Comfortable` (`8b-q4_K_M`, `3b*`); `Comfortable`–`Tight` (`8b-q8_0`); `Too Large` (`8b-bf16`, `30b`).
- **Est. M4 Decode Speed** `[Estimated]` (`~120 GB/s ÷ size_GB × 0.65–0.80`):
  - `8b-q4_K_M` (5.3 GB): **~14.7–18.1 tok/s**
  - `8b-q8_0` (9.3 GB): **~8.4–10.3 tok/s**
  - `3b-q4_K_M` (2.2 GB): **~35–44 tok/s**; `3b-q8_0` (3.9 GB): **~20–25 tok/s**
- **Recommended Ollama Env / Settings**: `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`; explicitly pass `temperature=0.0–0.2` in `pydantic-ai` (`ModelSettings`) because the default Modelfile injects `temperature=1.0, top_p=0.95` from `generation_config.json` `[Primary]`.

## Decision-Model Assessment

- **Decision Quality & Determinism** `[Primary]`: Standout enterprise instruction adherence (`IFBench 79.33` on 8B and `74.33` on 3B—the highest `<10B` IFBench scores in our repo), backed by IBM GRC governance and data clearance. Solid reasoning (`GPQA 64.14`, `MMLU-Pro 74.04`, `AIME25 86.67` on 8B). Override the default `temperature=1.0` to `0.0–0.2` for deterministic routing/decisions.
- **Structured Output & JSON Schema** `[Primary]`: All `<10B` tags are GGUF, so self-hosted Ollama enforces `json_schema` via `llama.cpp` grammar decoding for `pydantic-ai` `NativeOutput`.
- **Tool & Function Calling** `[Primary]`: Strong `τ³-bench` (`66.34` / `58.06` avg on 8B) and solid `BFCL v4` (`52.39` on 8B, `52.41` on 3B) with reasoning-augmented tool selection. **Operational check needed**: Ollama's web registry shows `[]` capability badges on `granite4.2` even though the README documents tool use and thinking—verify locally whether `/v1/chat/completions` accepts `tools` cleanly on `granite4.2:8b` (if Ollama rejects `tools`, patch via custom Modelfile `TEMPLATE` or use `NativeOutput`).
- **Thinking / Latency Tradeoff** `[Primary]`: Best-in-class control among generative models: `enable_thinking=True/False` plus `reasoning_effort="low"/"high"` (`/set nothink`, `/set think low|medium|high`, or REST `options.think`). Allows disabling `<think>` for low-latency classification or setting `low` effort for lightweight judgment.
- **Quantization & Context Notes** `[Primary]`: `RULER 64K / 128K` scores `80.99 / 71.41` on 8B (`67.52 / 55.30` on 3B). Dense 40-layer KV cache grows faster than hybrid DeltaNet models (`Qwen3.5`), so use `q8_0` KV cache at ≥16k context. 12 supported languages (EN, DE, ES, FR, JA, PT, AR, CS, IT, KO, NL, ZH).
- **Operational & pydantic-ai Readiness** `[Primary]`: **`Minor config`** via `OllamaModel` (`'ollama:granite4.2:8b-q4_K_M'`, `/v1`). Set `model_settings={'temperature': 0.0}` (overriding `temp=1.0` Modelfile default), configure `options.think` / `extra_body` to control `<think>` latency, and test `ToolOutput` vs. `NativeOutput` locally given the missing capability badge on Ollama's page.

## Benchmarks & Evidence

IBM reported benchmarks (`ibm-granite/granite-4.2-8b` & `3b` READMEs, NeMo Evaluator SDK, `temp=1.0, top_p=0.95`, `[Primary]`):

| Benchmark | `Granite-4.2-3B` | `Granite-4.2-8B` |
| --- | ---: | ---: |
| **IFBench (prompt)** | **74.33** | **79.33** |
| **τ³-bench / τ³-bench (AVG)** | 50.99 / 45.78 | **66.34 / 58.06** |
| **BFCL (v4)** | 52.41 | 52.39 |
| **MMLU-Pro** | 67.84 | 74.04 |
| **GPQA** | 54.80 | 64.14 |
| **LiveCodeBench v6** | 69.71 | 73.24 |
| **AIME25** | 78.33 | 86.67 |
| **SWE Bench Verified / Multilingual** | — | 47.67 / 30.78 |
| **RULER 64K / 128K** | 67.52 / 55.30 | **80.99 / 71.41** |

## Caveats & Open Questions

- **Missing Ollama Capability Badges**: Verify `granite4.2:8b` accepts OpenAI `/v1` `tools` payloads and `/set nothink` without needing a Modelfile override.
- **Default `temperature=1.0`**: Modelfile inherits `temperature=1.0` from `generation_config.json`; always override to `0.0–0.2` for deterministic decision pipelines.
- **Text-Only & Dense KV Cache**: No vision support; dense 40-layer GQA consumes ~160 KB/token fp16 KV cache on 8B (`5×` Qwen3.5-9B's 32 KB/tok).

## Sources

- `[2026-11-26]` [Ollama Library: granite4.2](https://ollama.com/library/granite4.2) & [granite4.2/tags](https://ollama.com/library/granite4.2/tags) — thinking parameters (`/set nothink`, `low/medium/high`), Modelfile `generation_config.json` behavior, tags.
- `[2026-11-26]` [Hugging Face: ibm-granite/granite-4.2-8b](https://huggingface.co/ibm-granite/granite-4.2-8b/raw/main/README.md) & [granite-4.2-3b](https://huggingface.co/ibm-granite/granite-4.2-3b/raw/main/README.md) — Apache-2.0 license, architecture table, `IFBench`/`τ³-bench`/`BFCL v4`/`RULER` benchmarks, Aug 25, 2026 release.
