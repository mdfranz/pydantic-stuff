# Liquid AI LFM3 1.7B (`lfm3`)

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `lfm3:1.7b-q8_0` (`a329e3b38ec2`), `lfm3:1.7b-mxfp8` (`4637931b9fcc`), `lfm3:1.7b-nvfp4` (`2012d88e202f`), `lfm3:1.7b-q4_K_M` (`b36a103fe2f8`), `lfm3:1.7b-qad-q4_0` (`765f7bd04f18`), `lfm3:1.7b-mlx-bf16` (`f39cc56813fb`), `lfm3:1.7b` / `lfm3:latest` (`df5fd9016d22`) |
| **Parameters (Total / Active)** | `1.73B` / `1.73B` |
| **Architecture** | Dense tri-hybrid (`Lfm3ForCausalLM`: 24 layers = 13 gated short convolutions + 5 sliding-window attention + 6 global `full_attention`) |
| **Context Window (Max / Practical M4)** | 256K (`262,144`) max / 8k–32k+ practical M4 *(evals at 32K tested by Liquid AI on Apple Silicon)* |
| **Recommended Quantization** | `Q8_0` (`lfm3:1.7b-q8_0`, 1.8 GB GGUF) for `NativeOutput` grammar guarantee; `MXFP8` (`lfm3:1.7b-mxfp8`, 2.1 GB MLX) or `MLX-BF16` (`lfm3:1.7b-mlx-bf16`, 3.5 GB) for Apple Silicon speed/zero loss |
| **License** | **LFM Open License v1.0** (free for research/non-commercial & commercial entities **< $10M annual revenue**; ≥ $10M commercial use is NOT licensed) |
| **pydantic-ai Readiness** | `Out-of-the-box` (`Minor config` — verify `<think>` & XML tool-call parsing / `NativeOutput` on MLX vs GGUF tags) |
| **Status** | `Shortlisted` *(Small generative tool-caller / structured-JSON / `ModelFallback` companion; NOT a `/v1/systemone` model)* |
| **Last Updated** | `2026-11-26` |

## Ollama Tags & Mac Mini M4 (24GB) Fit

| Ollama Tag | Hash | Disk Size | Backend | M4 24GB Fit | Notes |
| --- | --- | --- | --- | --- | --- |
| `lfm3:1.7b-qad-q4_0` | `765f7bd04f18` | 1.0 GB | GGUF | `Comfortable` | Smallest 4-bit QAD build |
| `lfm3:1.7b-q4_K_M` | `b36a103fe2f8` | 1.1 GB | GGUF | `Comfortable` | Default small GGUF (`lfm3:1.7b` / `latest` alias `1.1–1.7 GB`) |
| `lfm3:1.7b-nvfp4` | `2012d88e202f` | 1.7 GB | `MLX` | `Comfortable` | MLX 4-bit float |
| `lfm3:1.7b-q8_0` | `a329e3b38ec2` | 1.8 GB | GGUF | `Comfortable` | Near-lossless GGUF; `llama.cpp` grammar-constrained JSON |
| `lfm3:1.7b-mxfp8` | `4637931b9fcc` | 2.1 GB | `MLX` | `Comfortable` | Native Apple Silicon 8-bit MLX |
| `lfm3:1.7b-mlx-bf16` | `f39cc56813fb` | 3.5 GB | `MLX` | `Comfortable` | Full BF16 weights on MLX (`bf16` GGUF also 3.5 GB) |

- **Weight Size (Disk / Loaded)** `[Primary]`: `1.0–3.5 GB` across all tags (`1.8 GB` `q8_0`, `2.1 GB` `mxfp8`).
- **KV Cache Math** `[Estimated]`: Only 6 of 24 layers use global `full_attention` (`num_key_value_heads=4`, `head_dim=128`). At fp16, global KV cache is `6 × 2 × 4 × 128 × 2 B = 12 KB/token` (~96 MB @ 8k, ~192 MB @ 16k, ~384 MB @ 32k), plus tiny fixed convolution caches (`conv_L_cache=7`) and `sliding_window=256` buffers.
- **Est. Total RAM @ 8k / 16k / 32k** `[Estimated]` (weights + KV cache + ~0.6–0.8 GB Metal runtime):
  - `lfm3:1.7b-q4_K_M` (1.1 GB): **~1.8 GB** @ 8k / **~1.9 GB** @ 16k / **~2.1 GB** @ 32k (`Comfortable`).
  - `lfm3:1.7b-q8_0` (1.8 GB): **~2.5 GB** @ 8k / **~2.6 GB** @ 16k / **~2.8 GB** @ 32k (`Comfortable`).
  - `lfm3:1.7b-mxfp8` (2.1 GB): **~2.8 GB** @ 8k / **~2.9 GB** @ 16k / **~3.1 GB** @ 32k (`Comfortable`).
  - `lfm3:1.7b-mlx-bf16` (3.5 GB): **~4.3 GB** @ 8k / **~4.4 GB** @ 16k / **~4.6 GB** @ 32k (`Comfortable`).
- **Fit Tier**: `Comfortable` across every quantization (easily stays co-resident in 24GB RAM alongside `d3:9b` or `autotrust-jev:9b`).
- **Est. M4 Decode Speed** `[Estimated]` (`~120 GB/s ÷ size_GB × 0.65–0.80`):
  - `1.7b-q4_K_M` (1.1 GB): **~71–87 tok/s**
  - `1.7b-nvfp4` (1.7 GB): **~46–56 tok/s**
  - `1.7b-q8_0` (1.8 GB): **~43–53 tok/s**
  - `1.7b-mxfp8` (2.1 GB): **~37–46 tok/s**
  - `1.7b-mlx-bf16` (3.5 GB): **~22–27 tok/s**
- **Recommended Ollama Env / Settings**: General sampling recommended by Liquid AI: `temperature=0.1`, `top_k=50`, `repeat_penalty=1.1` `[Primary]`. Note: `lfm3` is a generative chat model (`tools`, `thinking`), not a `/v1/systemone` model.

## Decision-Model Assessment

- **Decision Quality & Determinism** `[Primary]`: Generative text model (`Lfm3ForCausalLM`), not a calibrated System One classifier (does not return native `noul`/`choice`/`score` probability distributions). Excels at instruction adherence (`IFEval 96.66`, `IFBench 73.03`) and abstention/non-hallucination (`AA-Omniscience non-hallucination 92.93%` vs. Qwen3.5-9B `52.59%`), though raw parametric knowledge accuracy is low (`7.07%` on AA-Omniscience) — Liquid AI explicitly recommends retrieval/search for knowledge-heavy tasks and advises against complex multi-step reasoning or agentic coding.
- **Structured Output & JSON Schema** `[Primary]`: Strong structured instruction adherence (`IFStruct 93.72` vs. Qwen3.5-9B `89.20`). On self-hosted Ollama, GGUF tags (`1.7b-q8_0`, `1.7b-q4_K_M`) leverage `llama.cpp` grammar-constrained decoding for `json_schema` (`NativeOutput`) `[Primary]`; verify locally whether `MLX` tags enforce `json_schema` identically.
- **Tool & Function Calling** `[Primary]`: Headline strength for a 1.7B model (`BFCL V4 71.39`, `ToolSandbox 80.36`, `τ³-bench 68.01`), outperforming Qwen3.5-9B (`62.83` / `76.41` / `67.11`) in Liquid AI's BF16 evals. Uses XML `<tool_call><function=...>` syntax in its Jinja template and supports deferred tool loading (`defer_loading=True`).
- **Thinking / Latency Tradeoff** `[Primary]`: **Critical caveat** — reasoning (`<think>`) is **enabled by default**, and Liquid AI explicitly warns that setting `enable_thinking=False` **"can be inconsistent, especially for reasoning-heavy tasks."** Even at ~45–80 tok/s on M4, default `<think>` traces add multi-second latency per call compared to single-pass System One models (`d3`, `autotrust-jev`).
- **Quantization & Context Notes** `[Primary]`: 16 languages supported (including EN, ZH, JA, KO, DE, FR, ES). 256K context window with `RULER 128K` `65.56` and `AA-LCR` `52.83` (BF16). At 1.8–3.5 GB for `Q8_0`/`MXFP8`/`MLX-BF16`, 4-bit quantization compromises are unnecessary on a 24GB M4 unless maximizing raw tok/s.
- **Operational & pydantic-ai Readiness** `[Primary]`: **`Out-of-the-box`** (`Minor config` recommended) via `OllamaModel` (`'ollama:lfm3:1.7b-q8_0'` or `'ollama:lfm3:1.7b-mxfp8'`) with `OLLAMA_BASE_URL='http://localhost:11434/v1'` (NOT `SystemOneModel`).
  - **Preferred output mode**: `NativeOutput` (using `lfm3:1.7b-q8_0` GGUF for `llama.cpp` `json_schema` enforcement) or `ToolOutput` (for agentic tool calling).
  - **Gaps to check locally**: (1) Confirm Ollama's `/v1/chat/completions` template reliably maps `lfm3`'s XML `<tool_call>` blocks into OpenAI `tool_calls` and separates `<think>` traces into `ThinkingPart` (if `<think>` leaks into `content` when using `PromptedOutput` or non-streaming `NativeOutput`, strip `<think>...</think>` or check `ModelProfile`); (2) verify whether `MLX` tags (`1.7b-mxfp8`) enforce `response_format: json_schema` as strictly as GGUF `1.7b-q8_0`.
  - **Recommended role**: Low-footprint (~2–3.5 GB) **System 2 / `ModelFallback` companion** paired behind a System One router (`d3:9b` or `autotrust-jev:9b`) for free-form `str` tool arguments and structured extraction.

## Benchmarks & Evidence

Liquid AI reported BF16 vLLM benchmarks (`LiquidAI/LFM3-1.7B` README) `[Primary]`:

- **Tool Use**: `BFCL V4` **`71.39`** (Qwen3.5-4B `55.12`, Qwen3.5-9B `62.83`); `ToolSandbox` **`80.36`** (Qwen3.5-9B `76.41`); `τ³-bench` (4 domains avg) **`68.01`** (MiniCPM5-2B `69.44`, Qwen3.5-9B `67.11`); `MCP-Atlas` `32.57` (Qwen3.5-9B `42.67`).
- **Instruction Following & Structure**: `IFEval` **`96.66`**; `IFBench` **`73.03`** (Qwen3.5-9B `48.84`); `IFStruct` **`93.72`** (Qwen3.5-9B `89.20`); `Multi-IF` **`89.10`**.
- **Agents**: `LiveClaw` **`45.60`**; `Claw-Eval General (EN)` avg **`68.64`** / `pass^3` `51.40`; `WebArena-Verified` **`38.42`**; `AutomationBench` **`41.29`**; `EnterpriseOps-Gym` `28.20`; `BrowseComp-Plus` `37.00`.
- **Knowledge & Abstention**: `AA-Omniscience non-hallucination` **`92.93%`** (accuracy `7.07%`, overall `0.50`); `GPQA Diamond` `63.37`.
- **Long Context**: `AA-LCR` `52.83`; `RULER 128K` `65.56`.
- **Edge Speed Claim** `[Primary]`: Liquid AI reports `LFM3-1.7B` (`Q4_0` GGUF in `llama.cpp` @ 32K ctx) prefills ≥ `2.5×` faster and decodes ≥ `1.8×` faster than `MiniCPM5-2B`, `Qwen3.5-4B`, `Qwen3.5-9B`, and `Gemma 4 E4B` on Apple M5 Max, AMD Ryzen AI Max+ 395, and iPhone 17 Pro *(caveat: Apple M5 Max numbers, not base Mac Mini M4)*.

## Caveats & Open Questions

- **Not a `/v1/systemone` Decision Model**: Unlike `d3` and `autotrust-jev`, `lfm3` generates text tokens and does not emit calibrated `noul`/`choice`/`score` probabilities natively (Liquid AI notes separate decision models exist in the LFM3 family, but none are published on Ollama yet).
- **License Threshold ($10M Revenue Cap)**: `LFM Open License v1.0` prohibits commercial use by legal entities with ≥ $10M annual revenue without a separate commercial agreement.
- **Default Thinking Overhead & Toggle Reliability**: `<think>` is on by default and `enable_thinking=False` is documented as inconsistent on harder prompts, which can hurt latency predictability in tight decision pipelines.
- **Small Parametric Knowledge**: Not suitable for closed-book factual reasoning or complex multi-step coding (`7.07%` raw AA-Omniscience accuracy); rely on context/RAG.
- **No Local M4 Measurements Yet**: Actual M4 tok/s, `<think>` suppression behavior in Ollama, and `pydantic-ai` `ToolOutput`/`NativeOutput` pass rates on `q8_0` vs. `mxfp8` need local verification.

## Sources

- `[2026-11-26]` [Ollama Library: lfm3](https://ollama.com/library/lfm3) & [lfm3/tags](https://ollama.com/library/lfm3/tags) — capabilities (`tools`, `thinking`), exact tags, sizes, GGUF + MLX variants.
- `[2026-11-26]` [Hugging Face: LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B/raw/main/README.md), [config.json](https://huggingface.co/LiquidAI/LFM3-1.7B/raw/main/config.json), & [LICENSE](https://huggingface.co/LiquidAI/LFM3-1.7B/raw/main/LICENSE) — tri-hybrid architecture, BF16 benchmarks, thinking/tool-call template details, LFM Open License v1.0 terms.
- `[2026-11-26]` [Pydantic AI Docs: Ollama](https://pydantic.dev/docs/ai/models/ollama/index.md) — `OllamaModel`, `NativeOutput` (`json_schema`), `ToolOutput`, and `PromptedOutput`.
