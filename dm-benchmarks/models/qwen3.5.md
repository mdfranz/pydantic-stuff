# Alibaba Qwen3.5 (`qwen3.5:4b` & `qwen3.5:9b`)

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `qwen3.5:9b-q4_K_M` (`56671c2ab938`), `9b-nvfp4` (`8218177ac78e`), `9b-mtp-nvfp4` (`a010df198560`), `9b-q8_0` (`a8ee95bc3592`), `9b-mxfp8` (`19c108424cec`), `qwen3.5:4b-q4_K_M` (`d8b0f5e9760c`), `4b-nvfp4` (`61aa3858e9d3`), `4b-q8_0` (`e15635ee9312`), `4b-mxfp8` (`77653d4d1b88`) *(plus `0.8b`, `2b`, and >10B `27b`/`35b`/`122b`)* |
| **Parameters (Total / Active)** | `9B` / `9B` · `4B` / `4B` (also `0.8B`, `2B`) — dense |
| **Architecture** | `Qwen3_5ForConditionalGeneration` hybrid (`32` layers: 24 Gated DeltaNet `linear_attention` + 8 `full_attention`, `num_key_value_heads=4`, `head_dim=256`, early-fusion vision) |
| **Context Window (Max / Practical M4)** | `256K` (`262,144`) max / `8k–32k` practical M4 |
| **Recommended Quantization** | `qwen3.5:9b-q4_K_M` (`6.6 GB`) / `9b-q8_0` (`10 GB`) GGUF for `NativeOutput`; `9b-nvfp4` (`7.2 GB`, MLX) / `9b-mtp-nvfp4` (`7.6 GB`, MLX) for speed; `4b-nvfp4` (`4.0 GB`) for co-residency |
| **License** | **Apache-2.0** |
| **pydantic-ai Readiness** | `Out-of-the-box` (`OllamaModel` `/v1`; configure `enable_thinking` toggle based on latency vs reasoning needs) |
| **Status** | `Shortlisted` (best general-purpose, 201-language Apache-2.0 `<10B` baseline; `ornith-1.5:9b` & `mimo-v2.6:9b` build on `9b`) |
| **Last Updated** | `2026-11-26` |

## Ollama Tags & Mac Mini M4 (24GB) Fit

Selected `<10B` tags (`Text, Image`, `256K` ctx, updated 1–2 months ago):

| Ollama Tag | Hash | Disk Size | Backend | M4 24GB Fit |
| --- | --- | ---: | --- | --- |
| `qwen3.5:0.8b` / `2b` | `6a061203ed02` / `0b28bbb3449d` | 1.0–3.1 GB | GGUF / `MLX` | `Comfortable` |
| `qwen3.5:4b-q4_K_M` | `d8b0f5e9760c` | 3.3 GB | GGUF | `Comfortable` |
| `qwen3.5:4b-nvfp4` (`4b-mlx`) | `61aa3858e9d3` | 4.0 GB | `MLX` | `Comfortable` |
| `qwen3.5:4b-q8_0` / `4b-mxfp8` | `e15635ee9312` / `77653d4d1b88` | 5.2 / 5.6 GB | GGUF / `MLX` | `Comfortable` |
| `qwen3.5:9b-q4_K_M` (`9b`, `latest`) | `56671c2ab938` / `fdf5fd4cd409` | **6.6 GB** | GGUF | `Comfortable` |
| `qwen3.5:9b-nvfp4` | `8218177ac78e` | **7.2 GB** | `MLX` | `Comfortable` |
| `qwen3.5:9b-mtp-nvfp4` | `a010df198560` | 7.6 GB | `MLX` (MTP) | `Comfortable` |
| `qwen3.5:9b-q8_0` | `a8ee95bc3592` | 10 GB | GGUF | `Comfortable` |
| `qwen3.5:9b-mxfp8` | `19c108424cec` | 11 GB | `MLX` | `Comfortable`–`Tight` |
| `qwen3.5:9b-bf16` / `9b-mlx-bf16` | `5c513fbbcfa8` / `1adfd61eea4f` | 19 GB | GGUF / `MLX` | `Too Large` |

- **Weight Size (Disk / Loaded)** `[Primary]`: `6.6–11 GB` (`9B`), `3.3–5.6 GB` (`4B`).
- **KV Cache Math** `[Estimated]`: Hybrid 3:1 Gated DeltaNet keeps fp16 KV cache to **~32 KB/token** on `9B` (~256 MB @ 8k, ~512 MB @ 16k, ~1.0 GB @ 32k).
- **Est. Total RAM @ 8k / 16k / 32k** `[Estimated]`:
  - `9b-q4_K_M` / `9b-nvfp4` (`6.6–7.2 GB`): **~7.8–9.2 GB** (`Comfortable`).
  - `9b-q8_0` (`10 GB`): **~11.2–12.2 GB** (`Comfortable`).
  - `4b-q4_K_M` / `4b-nvfp4` (`3.3–4.0 GB`): **~4.3–5.5 GB** (`Comfortable`).
- **Fit Tier**: `Comfortable` (`0.8b`, `2b`, `4b*`, `9b-q4_K_M`, `9b-nvfp4`, `9b-mtp-nvfp4`, `9b-q8_0`); `Comfortable`–`Tight` (`9b-mxfp8`); `Too Large` (`9b-bf16`, `27b`/`35b`/`122b`).
- **Est. M4 Decode Speed** `[Estimated]` (`~120 GB/s ÷ size_GB × 0.65–0.80`):
  - `9b-q4_K_M` (6.6 GB): **~11.8–14.5 tok/s**; `9b-nvfp4` (7.2 GB): **~10.8–13.3 tok/s** *(MTP tags can boost effective tok/s further)*.
  - `9b-q8_0` (10 GB): **~7.8–9.6 tok/s**.
  - `4b-q4_K_M` (3.3 GB): **~23.6–29.1 tok/s**; `4b-nvfp4` (4.0 GB): **~19.5–24.0 tok/s**.

## Decision-Model Assessment

- **Decision Quality & Determinism** `[Primary]`: The foundational Apache-2.0 `<10B` reference architecture (also the base of `autotrust-jev`, `d3-flash`, `kev`, `nimble-v3`, `mimo-v2.6`, and `ornith-1.5`). Strong balanced reasoning (`GPQA Diamond 80.4–81.7` on 9B, `75.8` on 4B), instruction adherence (`IFEval 92.1 / 92.5`, `IFStruct 89.2 / 87.1`), and **201 languages**. Note lower raw closed-book `AA-Omniscience non-hallucination` (`52.6%` on 9B, `13.6%` on 4B vs `92.9%` on `LFM3-1.7B`)—supply retrieved context rather than relying on parametric memory.
- **Structured Output & JSON Schema** `[Primary]`: `IFStruct 89.20` (9B) and `87.05` (4B). GGUF tags (`q4_K_M`, `q8_0`) apply `llama.cpp` `json_schema` grammar decoding for `pydantic-ai` `NativeOutput`.
- **Tool & Function Calling** `[Primary]`: Strong native tool use (`BFCL v4 62.8` on 9B, `BFCL v3 71.1` on 4B, `ToolSandbox 76.4 / 75.4`, `τ³-bench 67.1 / 66.0`).
- **Thinking / Latency Tradeoff** `[Primary]`: Operates in thinking mode (`<think>...</think>`) by default and cleanly supports disabling thinking (`enable_thinking=False`) for low-latency structured decisions. Also uniquely offers **MTP (`-mtp-`)** tags in Ollama (`9b-mtp-nvfp4` `7.6 GB`) for faster multi-token decoding.
- **Operational & pydantic-ai Readiness** `[Primary]`: **`Out-of-the-box`** via `OllamaModel` (`'ollama:qwen3.5:9b-q4_K_M'` or `'ollama:qwen3.5:4b-q4_K_M'`, `/v1`). Supports `NativeOutput`, `ToolOutput`, and clean thinking-tag toggling.

## Benchmarks & Evidence

Cross-verified BF16 figures (`Qwen3.5-9B` / `4B` in `LFM3-1.7B`, `LFM2.5-8B-A1B`, `MiMo-V2.6-9B`, and `Ornith-1.5-9B` HF tables, `[Primary]`):

| Benchmark | `Qwen3.5-4B` | `Qwen3.5-9B` |
| --- | ---: | ---: |
| **GPQA Diamond** | 75.76 | **80.40–81.7** |
| **IFEval** | **92.52** | 92.08 |
| **IFStruct / IFBench** | 87.05 / 38.16 | **89.20 / 48.84** |
| **BFCL v4 / ToolSandbox** | 54.01–55.12 / 75.40 | **62.83 / 76.41** |
| **BFCL v3** | **71.06** | — |
| **τ³-bench (avg) / MCP-Atlas** | 65.98 / 41.37 | **67.11 / 42.67** |
| **RULER 128K / AA-LCR** | 84.63 / 52.67 | **87.11 / 62.00** |
| **SWE-bench Verified** | 41.20 | **51.20–60.0** |
| **AA-Omniscience non-hallucination** | 13.64% | 52.59% |

## Caveats & Open Questions

- **Default Thinking Latency**: Thinking is enabled by default; disable thinking (`enable_thinking=False`) for fast classification/extraction so CoT doesn't stall latency.
- **Closed-Book Hallucination Rate**: Raw parametric non-hallucination rate (`52.6%` on 9B, `13.6%` on 4B) is much lower than `LFM3-1.7B` (`92.9%`)—always ground decisions in supplied context/RAG.
- **MTP & MLX `json_schema`**: Verify locally whether `-mtp-` and `-nvfp4` MLX tags enforce `NativeOutput` `json_schema` identically to GGUF tags on Ollama `0.40.18`.

## Sources

- `[2026-11-26]` [Ollama Library: qwen3.5](https://ollama.com/library/qwen3.5) & [qwen3.5/tags](https://ollama.com/library/qwen3.5/tags) — `0.8b`–`9b` tags (incl. `mtp` & `mlx`), `256K` ctx, capabilities (`vision`, `tools`, `thinking`).
- `[2026-11-26]` [Hugging Face: Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/README.md) & comparative HF tables (`LFM3-1.7B`, `LFM2.5-8B-A1B`, `MiMo-V2.6-9B`, `Ornith-1.5-9B`) — Apache-2.0 license, `enable_thinking` toggle, benchmark figures.
