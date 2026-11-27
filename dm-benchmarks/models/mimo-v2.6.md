# Xiaomi MiMo-V2.6 9B (`mimo-v2.6`)

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `mimo-v2.6:9b-q4_K_M` (`cc58eacf0e3e`), `mimo-v2.6:9b-nvfp4` (`13e6cddd37a6`), `mimo-v2.6:9b-q8_0` (`37f7310de102`), `mimo-v2.6:9b-mxfp8` (`74dc9c06c486`), `mimo-v2.6:9b` / `latest` (`4af06b564ed4`), plus `9b-mlx-bf16` / `9b-gguf-bf16` |
| **Parameters (Total / Active)** | `9B` / `9B` (dense) |
| **Architecture** | `Qwen/Qwen3.5-9B` hybrid (`32` layers: 24 Gated DeltaNet `linear_attention` + 8 `full_attention`, `num_key_value_heads=4`, `head_dim=256`, multimodal vision) |
| **Context Window (Max / Practical M4)** | `256K` (`262,144`) max / `8k–32k` practical M4 |
| **Recommended Quantization** | `mimo-v2.6:9b-q4_K_M` (`6.6 GB`, GGUF) or `mimo-v2.6:9b-q8_0` (`10 GB`, GGUF) for `llama.cpp` `NativeOutput` JSON schema; `mimo-v2.6:9b-nvfp4` (`8.7 GB`, MLX) for Apple Silicon speed |
| **License** | **MIT** *(HF card includes an informational acceptable-use reminder that explicitly does not amend the MIT license)* |
| **pydantic-ai Readiness** | `Out-of-the-box` (`Minor config`: verify `mimo` reasoning tag stripping & MLX `json_schema` vs GGUF) |
| **Status** | `Shortlisted` (MIT-licensed 9B multimodal agent/tool/security System 2 candidate) |
| **Last Updated** | `2026-11-26` |

## Ollama Tags & Mac Mini M4 (24GB) Fit

| Ollama Tag | Hash | Disk Size | Backend | Input | M4 24GB Fit |
| --- | --- | ---: | --- | --- | --- |
| `mimo-v2.6:9b-q4_K_M` | `cc58eacf0e3e` | 6.6 GB | GGUF | Text, Image | `Comfortable` |
| `mimo-v2.6:9b-nvfp4` | `13e6cddd37a6` | 8.7 GB | `MLX` | Text, Image | `Comfortable` |
| `mimo-v2.6:9b-q8_0` | `37f7310de102` | 10 GB | GGUF | Text, Image | `Comfortable` |
| `mimo-v2.6:9b-mxfp8` | `74dc9c06c486` | 11 GB | `MLX` | Text, Image | `Comfortable`–`Tight` |
| `mimo-v2.6:9b` / `latest` | `4af06b564ed4` | 6.6–8.7 GB | Auto | Text, Image | `Comfortable` |
| `mimo-v2.6:9b-mlx-bf16` / `gguf-bf16` | `98bfe7751130` / `08ec2ae01ade` | 19 GB | `MLX` / GGUF | Text, Image | `Too Large` |

- **Weight Size (Disk / Loaded)** `[Primary]`: `6.6 GB` (`q4_K_M`), `8.7 GB` (`nvfp4`), `10 GB` (`q8_0`), `11 GB` (`mxfp8`).
- **KV Cache Math** `[Estimated]`: Same `Qwen3.5-9B` 3:1 hybrid attention (~32 KB/token for fp16 KV cache: ~256 MB @ 8k, ~512 MB @ 16k, ~1.0 GB @ 32k).
- **Est. Total RAM @ 8k / 16k / 32k** `[Estimated]` (weights + KV cache + ~1.0–1.2 GB Metal overhead):
  - `mimo-v2.6:9b-q4_K_M` (6.6 GB): **~7.8–8.8 GB** (`Comfortable`).
  - `mimo-v2.6:9b-nvfp4` (8.7 GB): **~9.8–10.8 GB** (`Comfortable`).
  - `mimo-v2.6:9b-q8_0` (10 GB): **~11.2–12.2 GB** (`Comfortable`).
  - `mimo-v2.6:9b-mxfp8` (11 GB): **~12.2–13.2 GB** (`Comfortable`–`Tight`).
- **Fit Tier**: `Comfortable` (`q4_K_M`, `nvfp4`, `q8_0`); `Comfortable`–`Tight` (`mxfp8`); `Too Large` (`bf16` 19 GB tags).
- **Est. M4 Decode Speed** `[Estimated]` (`~120 GB/s ÷ size_GB × 0.65–0.80`):
  - `9b-q4_K_M` (6.6 GB): **~11.8–14.5 tok/s**
  - `9b-nvfp4` (8.7 GB): **~9.0–11.0 tok/s**
  - `9b-q8_0` (10 GB): **~7.8–9.6 tok/s**
  - `9b-mxfp8` (11 GB): **~7.1–8.7 tok/s**

## Decision-Model Assessment

- **Decision Quality & Determinism** `[Primary]`: Generative SFT distillation of `Qwen3.5-9B` on 77.4B MiMo tokens (27.2B loss-bearing: 29.9% Code, 28.5% General, 27.4% Visual, 14.2% Cyber). Big gains over base `Qwen3.5-9B` on agentic/automation benchmarks (`AutomationBench v1.0.6` `30.3` vs `5.0`, `Toolathlon-Verified` `35.2` vs `25.9`, `JobBench` `18.3` vs `2.6`). Note Xiaomi releases this checkpoint primarily as an **SFT starting point** for RL rather than a final RL-tuned model.
- **Structured Output & JSON Schema** `[Primary]`: On self-hosted Ollama, GGUF tags (`q4_K_M`, `q8_0`) apply `llama.cpp` grammar-constrained `json_schema` decoding for `pydantic-ai` `NativeOutput`. Verify locally whether `MLX` tags (`nvfp4`, `mxfp8`) enforce `json_schema` identically.
- **Tool & Function Calling** `[Primary]`: Ollama advertises `['vision', 'tools', 'thinking']`. Strong jump over `Qwen3.5-9B` on `Toolathlon-Verified` (`35.2` vs `25.9`) and `Terminal Bench 2.1` (`37.1` vs `27.0`).
- **Thinking / Latency Tradeoff** `[Primary]`: Supports `enable_thinking` toggle in its chat template (`chat_template_kwargs: {"enable_thinking": true/false}`), allowing fast no-think responses when desired or deep thinking on hard escalations. At ~9–14 tok/s on M4, unconstrained thinking traces can add several seconds per request.
- **Quantization & Context Notes** `[Primary]`: 256K max context; multimodal (`Text, Image`) across all tags, enabling visual coding/screenshot decisions.
- **Operational & pydantic-ai Readiness** `[Primary]`: **`Out-of-the-box`** (`Minor config`) via `OllamaModel` (`'ollama:mimo-v2.6:9b-q4_K_M'` or `'ollama:mimo-v2.6:9b-q8_0'`) with `OLLAMA_BASE_URL='http://localhost:11434/v1'`.
  - **Preferred output mode**: `NativeOutput` (GGUF grammar-constrained `json_schema`) or `ToolOutput`.
  - **Gaps to verify locally**: Confirm Ollama strips MiMo reasoning tags cleanly into `ThinkingPart` (avoiding pollution of `content` or `PromptedOutput` JSON), and compare `q4_K_M` vs `nvfp4` speed/schema fidelity on M4.

## Benchmarks & Evidence

Xiaomi reported SFT checkpoint results (`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` README, `[Primary]`):

| Benchmark | Metric | `Qwen3.5-9B` | `MiMo-V2.6-Distill-Qwen-9B` |
| --- | --- | ---: | ---: |
| SWE Verified | avg@3 | 60.0 | **61.1** |
| SWE Pro | avg@3 | 32.0 | **44.6** |
| AutomationBench v1.0.6 | avg@1 | 5.0 | **30.3** |
| Terminal Bench 2.1 | avg@1 | 27.0 | **37.1** |
| Toolathlon-Verified | avg@1 | 25.9 | **35.2** |
| OfficeQA | avg@1 | 9.0 | **19.5** |
| JobBench | avg@1 | 2.6 | **18.3** |
| MiMo Code / Cyber / General / Visual Coding (mini†) | avg@3 / avg@1 | 19.5 / 5.7 / 28.5 / 61.7 | **51.6 / 31.3 / 62.2 / 64.0** |

† Internal Xiaomi evaluation sets.

## Caveats & Open Questions

- **SFT-Only Checkpoint**: Released as an SFT base for RL (not post-RL like `Ornith-1.5-9B`), and Xiaomi does not publish `IFEval`/`BFCL` or probability calibration metrics in the model card.
- **Co-Residency RAM**: At `6.6–10 GB` (`q4_K_M` / `q8_0`), pairing `mimo-v2.6:9b` simultaneously in RAM with a 9B System One router (~8–10 GB) approaches ~16–19 GB (`Tight`); pair with `laya` (~1 GB) or `d3:4b` (~5.6 GB) if keeping both hot.
- **No Local M4 Measurements Yet**: Actual M4 tok/s, `enable_thinking` toggle reliability in Ollama, and `pydantic-ai` `NativeOutput`/`ToolOutput` accuracy are `TBD`.

## Sources

- `[2026-11-26]` [Ollama Library: mimo-v2.6](https://ollama.com/library/mimo-v2.6) & [mimo-v2.6/tags](https://ollama.com/library/mimo-v2.6/tags) — capabilities (`vision`, `tools`, `thinking`), all 8 tags, MIT license, headline eval table.
- `[2026-11-26]` [Hugging Face: XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/raw/main/README.md) — `Qwen3.5-9B` SFT mixture (77.4B tokens), full benchmark table, `enable_thinking` usage, MIT license.
