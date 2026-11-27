# Google Gemma 4 (`gemma4`)

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `gemma4:e4b` / `latest` (`009acb0d7fe1`), `e4b-it-qat` (`ee6656371218`), `e4b-it-q4_K_M` (`dc35e8d9c606`), `e4b-it-q8_0` (`3721c8128dc2`), `e4b-nvfp4` (`5a08735df03c`), `gemma4:e2b-it-qat` (`07ea59a47401`), `e2b-it-q4_K_M` (`b37049369adf`), `e2b-it-q8_0` (`8cb19ff6df84`) *(note: `12b`/`26b`/`31b` are >10B)* |
| **Parameters (Total / Active)** | `E4B`: ~8B total / 4B effective · `E2B`: ~5.1B total / 2B effective |
| **Architecture** | Dense multimodal (`Text`, `Image`, `Audio`) with configurable thinking & native system role |
| **Context Window (Max / Practical M4)** | `128K` (`E2B`/`E4B`; `256K` on `12B`/`26B`/`31B`) / `8k–32k` practical M4 |
| **Recommended Quantization** | `gemma4:e4b-it-qat` (`6.1 GB`) or `e4b-it-q4_K_M` (`6.6 GB`) for GGUF `NativeOutput`; `e4b-nvfp4` (`9.5 GB`, MLX) |
| **License** | **Apache-2.0** |
| **pydantic-ai Readiness** | `Out-of-the-box` (`Minor config`: configure thinking mode cleanly so `<think>` does not interfere with tool calls) |
| **Status** | `Shortlisted (Multimodal/Audio Niche)` — uniquely combines Apache-2.0 + `vision` + `audio` `<10B`, though trails `qwen3.5`/`granite4.2`/`lfm3` on tools |
| **Last Updated** | `2026-11-26` |

## Ollama Tags & Mac Mini M4 (24GB) Fit

| Ollama Tag | Hash | Disk Size | Backend | Input | M4 24GB Fit |
| --- | --- | ---: | --- | --- | --- |
| `gemma4:e2b-it-qat` / `q4_K_M` | `07ea59a47401` / `b37049369adf` | 4.3 / 4.6 GB | GGUF | Text, Image (+Audio) | `Comfortable` |
| `gemma4:e2b-it-q8_0` / `nvfp4` | `8cb19ff6df84` / `f4a8449d1781` | 6.0 / 7.5 GB | GGUF / `MLX` | Text, Image | `Comfortable` |
| `gemma4:e4b-it-qat` / `q4_K_M` | `ee6656371218` / `dc35e8d9c606` | **6.1 / 6.6 GB** | GGUF | Text, Image | `Comfortable` |
| `gemma4:e4b-it-q8_0` / `nvfp4` | `3721c8128dc2` / `5a08735df03c` | 9.0 / 9.5 GB | GGUF / `MLX` | Text, Image / `MLX` | `Comfortable` |
| `gemma4:e4b-mxfp8` | `17b5792c1f52` | 12 GB | `MLX` | Text, Image | `Tight` |
| `gemma4:e4b-bf16` | `c893c1170d92` | 16 GB | `MLX` / GGUF | Text, Image | `Tight` / Borderline `Too Large` |

*(Out-of-scope >10B siblings: `gemma4:12b-it-qat` `7.2 GB` fits 24GB M4 comfortably; `26b` `16–19 GB` is `Tight`/`Too Large`; `31b` `19–20 GB` is `Too Large`.)*

- **Weight Size (Disk / Loaded)** `[Primary]`: `6.1–6.6 GB` (`E4B QAT/Q4_K_M`), `9.0–9.5 GB` (`E4B Q8_0/NVFP4`), `4.3–4.6 GB` (`E2B QAT/Q4_K_M`).
- **Est. Total RAM @ 8k / 16k / 32k** `[Estimated]`: **~7.5–9.5 GB** (`E4B QAT/Q4_K_M`, `Comfortable`), **~10.5–12.5 GB** (`E4B Q8_0`, `Comfortable`–`Tight`).
- **Fit Tier**: `Comfortable` (`E2B`, `E4B QAT/Q4_K_M/Q8_0/NVFP4`); `Tight` (`E4B MXFP8`).
- **Est. M4 Decode Speed** `[Estimated]` (`~120 GB/s ÷ size_GB × 0.65–0.80`): **~12.0–15.7 tok/s** (`E4B QAT/Q4_K_M`), **~8.2–10.7 tok/s** (`E4B Q8_0/NVFP4`), **~17–22 tok/s** (`E2B QAT/Q4_K_M`).

## Decision-Model Assessment

- **Decision Quality & Determinism** `[Primary]`: Big leap over Gemma 3, and now **Apache-2.0** with native system-prompt support (`Gemma 4 E4B`: `MMLU-Pro 69.4%`, `GPQA Diamond 58.6%`, `IFEval 87.74`, `Tau2 avg 42.2%`). However, independent/comparative tables (`LFM2.5` & `LFM3` HF cards) show `E4B` trailing `Qwen3.5-4B/9B`, `Granite-4.2-8B`, and `LFM3-1.7B` on `IFBench` (`39.48`), `BFCL v4` (`33.92`), and `LiveCodeBench` (`52.0`).
- **Structured Output & JSON Schema** `[Primary]`: GGUF tags (`e4b-it-qat`, `e4b-it-q4_K_M`, `e4b-it-q8_0`) enforce `json_schema` via `llama.cpp` grammar decoding for `pydantic-ai` `NativeOutput`.
- **Tool & Function Calling** `[Primary]`: Native function calling (`BFCL v3 57.31`, `BFCL v4 33.92`, `Tau2 avg 42.2%` on `E4B`). Trails `qwen3.5:4b/9b` and `lfm3:1.7b` on complex tool benchmarks.
- **Thinking / Latency Tradeoff** `[Primary]`: Configurable thinking mode; disable thinking for fast structured extraction or enable for math/reasoning (`AIME 2026 42.5%` on `E4B`).
- **Multimodal Advantage** `[Primary]`: Ollama advertises `['vision', 'tools', 'thinking', 'audio']`—uniquely includes **native audio + variable-aspect-ratio image** capabilities under 10B with Apache-2.0.
- **Operational & pydantic-ai Readiness** `[Primary]`: **`Out-of-the-box`** (`Minor config`) via `OllamaModel` (`'ollama:gemma4:e4b-it-qat'` or `'ollama:gemma4:e4b-it-q4_K_M'`, `/v1`). Configure thinking toggle cleanly so reasoning blocks don't pollute `ToolOutput`/`PromptedOutput`.

## Benchmarks & Evidence

- **Ollama / Google Gemma 4 Table (`E4B` / `E2B`)** `[Primary]`:
  - `MMLU-Pro`: `69.4%` / `60.0%`
  - `GPQA Diamond`: `58.6%` / `43.4%`
  - `LiveCodeBench v6`: `52.0%` / `44.0%`
  - `AIME 2026 (no tools)`: `42.5%` / `37.5%`
  - `Tau2 (average over 3)`: `42.2%` / `24.5%`
- **Cross-Model HF Comparisons (`LFM2.5` & `LFM3` cards, `E4B` / `E2B`)** `[Primary]`:
  - `IFEval`: `87.74` / `82.93`; `IFBench`: `39.48` / `33.53`; `Multi-IF`: `77.58` / `69.70`
  - `BFCL v3`: `57.31` / `56.44`; `BFCL v4`: `33.92` / `31.91`
  - `AA-Omniscience non-hallucination`: `36.06%` / `15.05%`

## Caveats & Open Questions

- **Weaker Tool Calling & `IFBench` vs. `<10B` Peers**: `BFCL v4` (`33.92`) and `IFBench` (`39.48`) lag `qwen3.5:4b/9b`, `granite4.2:8b`, and `lfm3:1.7b`.
- **Audio Input in Ollama**: While Ollama lists the `audio` capability badge on `gemma4`, `/tags` currently lists `Text, Image`—verify audio input support in Ollama `/v1` if needed.
- **Note on `gemma4:12b`**: `12b` (`7.7–8.0 GB`, `12b-it-qat` `7.2 GB`) fits 24GB M4 comfortably if you ever relax the `<10B` parameter filter.

## Sources

- `[2026-11-26]` [Ollama Library: gemma4](https://ollama.com/library/gemma4) & [gemma4/tags](https://ollama.com/library/gemma4/tags) — `E2B`/`E4B` specs, capabilities (`vision`, `tools`, `thinking`, `audio`), benchmark table, all tags.
- `[2026-11-26]` [Hugging Face: LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/raw/main/README.md) & [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B/raw/main/README.md) — comparative `IFEval`, `IFBench`, `BFCL`, and `AA-Omniscience` figures for `Gemma-4-E2B/E4B-IT`.
