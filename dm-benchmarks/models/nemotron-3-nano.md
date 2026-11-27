# NVIDIA Nemotron 3 Nano 4B (`nemotron-3-nano:4b`)

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `nemotron-3-nano:4b` (`28e2e9e36b77`), `4b-q4_K_M` (`6cc467f05439`), `4b-nvfp4` / `4b-mlx` (`87e6cab38edd`), `4b-mxfp8` (`5eab47118f52`), `4b-q8_0` (`c534497d032c`), `4b-mlx-bf16` (`2b68ba060a87`) *(WARNING: `:latest` pulls **`30b`** `18–24 GB`!)* |
| **Parameters (Total / Active)** | `4B` / `4B` (dense compressed hybrid; `30b` is `30B / 3.5B` active — >10B & `Too Large`) |
| **Architecture** | Mamba-2 + Transformer Hybrid (`Text`-only) |
| **Context Window (Max / Practical M4)** | `256K` (`4B`; `1M` on `30B`) / `16k–64k+` practical M4 (`Mamba-2` reduces KV growth) |
| **Recommended Quantization** | `nemotron-3-nano:4b-q4_K_M` (`2.8 GB`, GGUF) or `4b-q8_0` (`4.2 GB`, GGUF) for `NativeOutput` grammar; `4b-nvfp4` (`2.9 GB`, MLX) / `4b-mxfp8` (`4.1 GB`, MLX) for Apple Silicon speed |
| **License** | **NVIDIA Open Model License Agreement** (permissive commercial open model license, non-OSI) |
| **pydantic-ai Readiness** | `Out-of-the-box` (`OllamaModel` `/v1`; supports `enable_thinking=False` toggle) |
| **Status** | `Shortlisted` (top `<5B` long-context `RULER 128K 91.1` & toggleable-thinking text tool/decision model) |
| **Last Updated** | `2026-11-26` |

## Ollama Tags & Mac Mini M4 (24GB) Fit

| Ollama Tag | Hash | Disk Size | Backend | Ctx | M4 24GB Fit |
| --- | --- | ---: | --- | ---: | --- |
| `nemotron-3-nano:4b-q4_K_M` | `6cc467f05439` | 2.8 GB | GGUF | 256K | `Comfortable` |
| `nemotron-3-nano:4b-nvfp4` / `4b-mlx` | `87e6cab38edd` | 2.9 GB | `MLX` | 256K | `Comfortable` |
| `nemotron-3-nano:4b-mxfp8` | `5eab47118f52` | 4.1 GB | `MLX` | 256K | `Comfortable` |
| `nemotron-3-nano:4b-q8_0` | `c534497d032c` | 4.2 GB | GGUF | 256K | `Comfortable` |
| `nemotron-3-nano:4b-mlx-bf16` / `4b-bf16` | `2b68ba060a87` / `4bc6e34d03fb` | 8.0 GB | `MLX` / GGUF | 256K | `Comfortable` |
| `nemotron-3-nano:30b` (`latest`) / `30b-a3b-nvfp4` | `373f79c3ce94` / `c436912ee99d` | 18–24 GB | `MLX` / GGUF | 256K–1M | `Too Large` *(>10B & >18 GB loaded)* |

- **Weight Size (Disk / Loaded)** `[Primary]`: `2.8–4.2 GB` (`4b-q4_K_M`, `4b-nvfp4`, `4b-mxfp8`, `4b-q8_0`), `8.0 GB` (`4b-mlx-bf16`).
- **Est. Total RAM @ 8k / 16k / 32k** `[Estimated]` (Mamba-2 hybrid keeps KV cache small): **~3.8–4.6 GB** (`4b-q4_K_M` / `4b-nvfp4`), **~5.2–6.0 GB** (`4b-mxfp8` / `4b-q8_0`) (`Comfortable`).
- **Fit Tier**: `Comfortable` (all `4b` tags); `Too Large` (`30b` / `latest`).
- **Est. M4 Decode Speed** `[Estimated]` (`~120 GB/s ÷ size_GB × 0.65–0.80`):
  - `4b-q4_K_M` / `4b-nvfp4` (`2.8–2.9 GB`): **~27–34 tok/s**
  - `4b-mxfp8` / `4b-q8_0` (`4.1–4.2 GB`): **~18.5–23.4 tok/s**
  - `4b-mlx-bf16` (`8.0 GB`): **~9.8–12.0 tok/s**

## Decision-Model Assessment

- **Decision Quality & Determinism** `[Primary]`: Strong `<5B` reasoning and long-context retention (`RULER 128K 91.1`, beating `Qwen3.5-9B`'s `87.1` and `Granite-4.2-8B`'s `71.4`!). Strong instruction adherence (`IFEval-Instruction 88.0–92.0`, `Prompt 82.8–87.9`) and math (`MATH500 95.4`, `AIME25 78.5`).
- **Structured Output & JSON Schema** `[Primary]`: GGUF tags (`4b-q4_K_M`, `4b-q8_0`) use `llama.cpp` grammar-constrained `json_schema` decoding for `pydantic-ai` `NativeOutput`.
- **Tool & Function Calling** `[Primary]`: Ollama lists `['tools', 'thinking']`. `BFCL v3` scores `61.1`, and `Tau2` scores `33.3` (Airline) / `39.8` (Retail) / `33.0` (Telecom) in BF16 evals (`4B`).
- **Thinking / Latency Tradeoff** `[Primary]`: Unified reasoning + non-reasoning model (`enable_thinking=True` by default; supports `enable_thinking=False` to skip `<think>` traces for fast direct decisions with a slight accuracy drop on hard prompts).
- **Quantization & Context Notes** `[Primary]`: Text-only (`256K` ctx). Languages: EN, DE, ES, FR, IT, JA. **Tag pitfall**: `ollama run nemotron-3-nano` (`:latest`) pulls **`30b`** (`18–24 GB`), not `4b`—always specify `:4b`.
- **Operational & pydantic-ai Readiness** `[Primary]`: **`Out-of-the-box`** via `OllamaModel` (`'ollama:nemotron-3-nano:4b-q4_K_M'` or `'ollama:nemotron-3-nano:4b-q8_0'`, `/v1`). Supports `NativeOutput`, `ToolOutput`, and `enable_thinking=False`.

## Benchmarks & Evidence

NVIDIA reported benchmarks (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` README, `[Primary]`):

- **Long Context**: `RULER (128k)` **`91.1`**
- **Instruction Following**: `IFEval-Instruction` **`88.0–92.0`** | `IFEval-Prompt` `82.8–87.9` | `IFBench-Prompt / Instruction` `43.2 / 44.2`
- **Tool Calling**: `BFCL v3` **`61.1`** | `Tau2` (Airline / Retail / Telecom): `28.0–33.3` / `34.8–39.8` / `24.9–33.0`
- **Reasoning & Coding**: `MATH500` **`95.4`** | `AIME25` **`78.5`** | `GPQA` `53.2` | `LiveCodeBench` `51.8` | `HaluEval` `62.2`

## Caveats & Open Questions

- **`:latest` Tag Trap**: `nemotron-3-nano:latest` points to `30b` (`18–24 GB`, `Too Large`); explicitly use `nemotron-3-nano:4b-*`.
- **Text-Only & Non-OSI License**: No vision support; governed by the **NVIDIA Open Model License Agreement** (commercial use allowed under NVIDIA terms).
- **No Local M4 Measurements Yet**: M4 tok/s, `NativeOutput` / `ToolOutput` schema reliability, and `enable_thinking=False` latency/accuracy trade-off are `TBD`.

## Sources

- `[2026-11-26]` [Ollama Library: nemotron-3-nano](https://ollama.com/library/nemotron-3-nano) & [nemotron-3-nano/tags](https://ollama.com/library/nemotron-3-nano/tags) — `4b` vs `30b` tags, capabilities (`tools`, `thinking`), NVIDIA Open Model License.
- `[2026-11-26]` [Hugging Face: nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/raw/main/README.md) — Mamba-2 + Transformer hybrid, `RULER 128k (91.1)`, `BFCL v3 (61.1)`, `IFEval`, `enable_thinking` toggle.
