# Convai Innovations Laya (`laya`)

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `laya:latest` (`0175aca9f021`), `laya:421m-typed-decisions-bf16` (`84a7be5dd978`), `laya:421m-typed-decisions-mlx-fp16` (`c62bc110244c`), `laya:421m-english-bf16` (`1f196b378202`), `laya:421m-english-mlx-fp16` (`aaba5574a1e1`), `laya:322m-multilingual-bf16` (`c1242e4a83f0`), `laya:322m-multilingual-mlx-fp16` (`63b6e37ecedb`) |
| **Parameters (Total / Active)** | `421M` (`ModernBERT-large`) · `322M` (`mmBERT-base`) |
| **Architecture** | Non-autoregressive encoder-only (`ModernBERT-large` / `mmBERT-base`), `[MASK]`-token option scoring trained via RLCD against strictly proper scoring rules |
| **Context Window (Max / Practical M4)** | **CRITICAL tag difference**: `-bf16` GGUF tags & `laya:latest` = **`8K`**; `-mlx-fp16` tags = **`1K`** (`typed-decisions`, `multilingual`) or **`512`** (`english`) `[Primary]` |
| **Recommended Quantization** | Unquantized `BF16` / `FP16` (`658–846 MB`) — tiny enough that 4-bit/8-bit quantization is unnecessary |
| **License** | **Apache-2.0** for weights & `laya` package *(caveat: optional continuous-learning `vegaml-learn` package is PolyForm Noncommercial 1.0.0)* |
| **pydantic-ai Readiness** | `Out-of-the-box` (`Minor config`: set `decision_max_choice_options=20`, and match `context_window` to chosen tag) |
| **Status** | `Shortlisted` (fastest sub-1GB Apache-2.0 router & 100+ language System 1 classifier) |
| **Last Updated** | `2026-11-26` |

## Ollama Tags & Mac Mini M4 (24GB) Fit

Requires **Ollama v0.40.0 or later** `[Primary]`. Notice that the `MLX` tags carry much shorter context limits (`512` or `1K`) than the `bf16` tags (`8K`):

| Ollama Tag | Hash | Disk Size | Ctx | Input / Languages | Backend | M4 24GB Fit |
| --- | --- | ---: | ---: | --- | --- | --- |
| `laya:latest` | `0175aca9f021` | 844–846 MB | **8K** | Text | Auto | `Comfortable` |
| `laya:421m-typed-decisions-bf16` | `84a7be5dd978` | 844 MB | **8K** | Text (EN workflows) | GGUF | `Comfortable` |
| `laya:421m-typed-decisions-mlx-fp16` | `c62bc110244c` | 846 MB | **1K** | Text (EN workflows) | `MLX` | `Comfortable` |
| `laya:421m-english-bf16` | `1f196b378202` | 844 MB | **8K** | Text (EN general) | GGUF | `Comfortable` |
| `laya:421m-english-mlx-fp16` | `aaba5574a1e1` | 846 MB | **512** | Text (EN general) | `MLX` | `Comfortable` |
| `laya:322m-multilingual-bf16` | `c1242e4a83f0` | 658 MB | **8K** | Text (100+ languages) | GGUF | `Comfortable` |
| `laya:322m-multilingual-mlx-fp16` | `63b6e37ecedb` | 678 MB | **1K** | Text (100+ languages) | `MLX` | `Comfortable` |

- **Weight Size (Disk / Loaded)** `[Primary]`: `658–846 MB` (< 0.9 GB).
- **Est. Total RAM @ 1k / 8k** `[Estimated]`: **~1.0–1.5 GB** total RAM (`Comfortable` — negligible memory footprint; easily kept permanently hot alongside any LLM).
- **Fit Tier**: `Comfortable` across all variants.
- **Est. M4 Decode Speed** `[Primary]` `[Estimated]`: Non-autoregressive single forward pass; Ollama reports **< 10 ms** for 1 question on M5 Max `[Primary]` (and `32.8–39.5 ms` on a T4 GPU, `72.3 ms` for 10 batched questions `[Primary]`). Expect **~10–30 ms** per single-pass request on base M4 `[Estimated]`.

## Decision-Model Assessment

- **Decision Quality & Determinism** `[Primary]`: Built for high-speed, deterministic triage (email spam/phishing/routing, content safety, intent routing, fact verification, agent observability, invoices, security incidents). Trained via RLCD against strictly proper scoring rules; in-task calibration error `ECE = 0.060` (post-temperature `0.081` vs. raw `0.213`). Up to **64 questions** share a single forward pass on `/v1/systemone`.
- **Structured Output & JSON Schema** `[Primary]`: Native `/v1/systemone` (`choice`, `noul`, `score`). **Option-count limit**: all options share a fixed `192–256` token head budget, so Convai explicitly advises **keeping `choice` questions under ~20 options** (accuracy on Banking77 with 77 labels drops to `42.5%`). **Ordinal `score` is the weakest primitive** (`SST-5` `0.372`).
- **Tool & Function Calling** `[Primary]`: `/v1/systemone` router only (ideal as a sub-20ms pre-router before a larger `pydantic-ai` LLM).
- **Thinking / Latency Tradeoff** `[Primary]`: Encoder-only (`ModernBERT` / `mmBERT`); zero decoder, zero text generation, lowest latency in this repository.
- **Quantization & Context Notes** `[Primary]`: Check tag context carefully—`laya:421m-english-mlx-fp16` is capped at `512` tokens and `mlx-fp16` `typed-decisions`/`multilingual` are capped at `1K`, whereas `-bf16` tags and `laya:latest` support `8K`. `322m-multilingual` uses `mmBERT-base` for **100+ languages**.
- **Operational & pydantic-ai Readiness** `[Primary]`: **`Out-of-the-box`** (`Minor config` recommended) via `SystemOneModel` (`'system-one:laya'` or `'system-one:laya:421m-typed-decisions-bf16'`) + `SystemOneProvider(base_url='http://localhost:11434')` + `DecisionModelProfile(decision_max_choice_options=20, context_window=8192)` (or `1024`/`512` for `-mlx-fp16` tags).

## Benchmarks & Evidence

- **Ollama & HF Benchmarks (`convaiinnovations/laya` & `laya-typed-decisions`)** `[Primary]`:
  - `laya` (421M): In-task `ECE = 0.060`; post-temp `ECE = 0.081` (`3×` lower than reported baseline `0.246`); `32.8 ms` p50 (1 question, T4) / `72.3 ms` (10 questions batched); `< 10 ms` on Apple M5 Max `[Primary]`.
  - `laya-typed-decisions` (421M): `76.6%` accuracy across the four typed-decisions workflows (agent observability, customer service, invoice processing, security incidents).
  - High-cardinality limit: Banking77 accuracy drops from `~87%` (with post-tuning/fewer labels) to `42.5%` at 77 labels due to shared option token budget.
  - Ordinal `score` weakness: `SST-5` `0.372`.

## Caveats & Open Questions

- **Text-Only on Ollama**: While newer Laya 0.5 Python packages (`vegaml` / reader-based) mention multimodal/73K context, Ollama's `laya` tags are **Text-only** (`421M`/`322M` encoders) with `512` / `1K` / `8K` context caps.
- **Weak on > ~20 Options & Ordinal `score`**: Use `kev:9b-nvfp4` or `d3` for >20 options or nuanced rubric scoring; use `laya` for fast routing, binary guardrails (`noul`), and ≤20-option triage.
- **Raw Calibration Before Temp Scaling**: Out-of-the-box pre-temperature ECE (`0.213`) is higher than post-domain scaling (`0.081`); validate thresholds on your own data.
- **Continuous Learning License**: Weights and `laya` inference are **Apache-2.0**, while optional `vegaml-learn` is noncommercial (`PolyForm Noncommercial 1.0.0`).

## Sources

- `[2026-11-26]` [Ollama Library: laya](https://ollama.com/library/laya) & [laya/tags](https://ollama.com/library/laya/tags) — v0.40.0+ requirement, `<10 ms` M5 Max claim, `ECE 0.060`, exact tags & `512`/`1K`/`8K` context differences.
- `[2026-11-26]` [Hugging Face: convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya/raw/main/README.md) & [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions/raw/main/README.md) — `ModernBERT-large` / `mmBERT-base` encoders, RLCD, Apache-2.0 weights vs `vegaml-learn` license, `~20`-option and `SST-5` limits.
