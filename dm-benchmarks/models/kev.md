# Kev (`kev`)

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `kev:9b-nvfp4` / `kev:latest` (`d31165461ae8` / `079f29347a00`), `kev:9b-q4_K_M` (`96375189a97f`), `kev:9b-q8_0` (`446718ead956`), `kev:4b-q4_K_M` (`818b6cdb957c`), `kev:4b-q8_0` (`46203beabf65`), `kev:0.8b-mxfp8` (`e414a229c2b8`), plus `mlx-bf16` / `bf16` / `27b*` |
| **Parameters (Total / Active)** | `0.8B` · `4B` · `9B` · `27B` (dense) |
| **Architecture** | `Qwen3.5-{0.8B,4B,9B}-Base` (24 Gated DeltaNet + 8 full attention on 9B/4B) & `Qwen3.8-27B` with LoRA + FP32 pointer head |
| **Context Window (Max / Practical M4)** | `73,728` tag `num_ctx` (`256K` in registry) / **8,192 validated** (`kev-4b` & `kev-9b` degrade >3 pp at 16k `[Primary]`) |
| **Recommended Quantization** | `kev:9b-nvfp4` (`6.7 GB`, mixed NVFP4/MXFP8 MLX, `kev:latest`); `kev:4b-q8_0` (`4.5 GB`) for smaller footprint |
| **License** | Apache-2.0 |
| **pydantic-ai Readiness** | `Out-of-the-box` (`Minor config`: set `decision_max_choice_options=255`, `decision_max_score_levels=255`, `context_window=8192`) |
| **Status** | `Shortlisted` (`kev:9b-nvfp4` — top Apache-2.0 text classifier supporting up to 255 options) |
| **Last Updated** | `2026-11-26` |

## Ollama Tags, Resident Memory & Mac Mini M4 (24GB) Fit

Requires **Ollama v0.40.4 or later** `[Primary]`. Resident memory & JevBench v1.2.16 results below are primary measurements reported on the Ollama page (one Apple M5 Max 128 GB, `/api/ps` after 3-question workload):

| Ollama Tag | Format / Hash | Disk Size | M5 Max Resident `[Primary]` | Public JevBench (231) / Hard (111) `[Primary]` | M4 24GB Fit |
| --- | --- | --- | ---: | ---: | --- |
| `kev:0.8b-mxfp8` | MLX MXFP8 (`e414a229c2b8`) | 1.0 GB | 2.84 GiB | 146/231 (`39/111`) | `Comfortable` |
| `kev:0.8b-mlx-bf16` | MLX BF16 (`f8bb3dfed81d`) | 1.5 GB | 3.28 GiB | 146/231 (`39/111`) | `Comfortable` |
| `kev:4b-q4_K_M` / `4b-q8_0` | GGUF (`818b6cdb957c` / `46203beabf65`) | 2.7 / 4.5 GB | ~3.6 / ~5.5 GiB `[Est]` | — | `Comfortable` |
| `kev:4b-mlx-bf16` | MLX BF16 (`458360f7747d`) | 8.4 GB | 12.47 GiB | 175/231 (`60/111`) | `Tight` |
| `kev:9b-q4_K_M` | GGUF (`96375189a97f`) | 5.6 GB | ~6.8 GiB `[Est]` | — | `Comfortable` |
| `kev:9b-nvfp4` (`kev:latest`) | Mixed NVFP4/MXFP8 (`d31165461ae8`) | **6.7 GB** | **11.16 GiB** | **183/231 (`68/111`)** | `Comfortable` |
| `kev:9b-q8_0` | GGUF (`446718ead956`) | 9.5 GB | ~11.0 GiB `[Est]` | — | `Comfortable` |
| `kev:9b-mlx-bf16` / `9b-bf16` | BF16 (`7044de0996a2` / `925e229ef61d`) | 16 / 18 GB | 19.59 GiB | 184/231 (`66/111`) | `Too Large` |
| `kev:27b-nvfp4` | Mixed NVFP4/MXFP8 (`4f5575ee2df2`) | 19 GB | 25.56 GiB | 205/231 (`86/111`) | `Too Large` |
| `kev:27b-mlx-bf16` | MLX BF16 (`b397feb9fd33`) | 51 GB | 55.56 GiB | 202/231 (`83/111`) | `Too Large` |

- **Weight Size (Disk / Loaded)**: `kev:9b-nvfp4` (`6.7 GB` disk, **11.16 GiB** resident on Apple Silicon `[Primary]`).
- **Est. Total RAM @ 8k / 16k / 32k** `[Estimated]`: `kev:9b-nvfp4` uses ~11.2 GiB at low context (matching `/api/ps` 11.16 GiB) to ~11.7–12.2 GiB @ 8k–16k (`Comfortable`), leaving solid OS headroom on 24GB.
- **Fit Tier**: `Comfortable` (`9b-nvfp4`, `9b-q4_K_M`, `9b-q8_0`, `4b-q4_K_M`/`q8_0`, `0.8b*`); `Tight` (`4b-mlx-bf16`); `Too Large` (`9b-mlx-bf16`, `27b*`).
- **Est. M4 Decode Speed** `[Estimated]`: Single forward pass (0 output tokens). M4 120 GB/s weight-read floor for `9b-nvfp4` (6.7 GB) is ~56 ms; expect ~130–350 ms per text decision (~11.6–14.3 tok/s eq. decode bandwidth).

## Decision-Model Assessment

- **Decision Quality & Determinism** `[Primary]`: Strong Apache-2.0 text decision model with transparent out-of-domain & calibration evals. Notably, `kev:9b-nvfp4` scored **`183/231`** overall and **`68/111`** on the hard JevBench subset—outperforming `kev:4b-mlx-bf16` (`60/111`) while using *less* resident memory (`11.16 GiB` vs `12.47 GiB`). `0.8b` struggles on hard reasoning (`39/111`).
- **Structured Output & JSON Schema** `[Primary]`: Native `/v1/systemone` endpoint (`choice`, `noul`, `score`). Standout capability: **accepts 1–255 named `choice` options and 1–255 ordered `score` levels** on Ollama (pointer head remains FP32 in every quant format), bypassing the 26-option ceiling of standard verbalizer-slot models.
- **Tool & Function Calling** `[Primary]`: `/v1/systemone` only; pair with `pydantic-ai` `DecisionModel` + `ModelFallback` for tool execution/free-form arguments.
- **Thinking / Latency Tradeoff** `[Primary]`: Prefill-only single pass; zero thinking-token generation.
- **Quantization & Context Notes** `[Primary]`: Tag `num_ctx` is `73,728` tokens, **but publisher validation stops at `8,192` tokens** (both `kev-4b` and `kev-9b` fail the −3 pp tolerance at 16k on CUAD). English-focused evaluations; option order can still shift answers despite question isolation.
- **Operational & pydantic-ai Readiness** `[Primary]`: **`Out-of-the-box`** (`Minor config` recommended) via `SystemOneModel` (`'system-one:kev:9b-nvfp4'`) + `SystemOneProvider(base_url='http://localhost:11434')` + `DecisionModelProfile(decision_max_choice_options=255, decision_max_score_levels=255, context_window=8192)`.

## Benchmarks & Evidence

- **Ollama Local JevBench v1.2.16 (231 public tasks / 111 hard tasks, Apple M5 Max)** `[Primary]`:
  - `kev:9b-nvfp4`: **`183/231`** (`68/111` hard), `11.16 GiB` resident.
  - `kev:9b-mlx-bf16`: `184/231` (`66/111` hard), `19.59 GiB` resident.
  - `kev:4b-mlx-bf16`: `175/231` (`60/111` hard), `12.47 GiB` resident.
  - `kev:0.8b-mxfp8`: `146/231` (`39/111` hard), `2.84 GiB` resident.
- **Publisher Evaluations (`jaredpalmer/kev-9b` & `kev-4b` HF cards, unquantized)** `[Primary]`:
  - `transfer-v4` OOD locked test (656 questions): `kev-9b` **`85.2%`** acc, `0.199` Brier, **`0.034` ECE**, `1.4%` confident errors (p ≥ 0.9), `74.2%` coverage at ≤ 5% error (`kev-4b`: `83.8%` acc, `0.224` Brier, `0.017` ECE, `1.5%` confident errors).
  - `breadth-v1` (14 held-out public datasets, 3,089 questions): `kev-9b` `69.8%`, `kev-4b` `69.0%`.
  - `MMLU-Pro` (10 options): `kev-9b` `59.0%`, `kev-4b` `56.5%`.

## Caveats & Open Questions

- **Text-Only**: No image/vision support (unlike `d3` and `d1`).
- **8,192-Token Validated Ceiling**: Do not rely on the `73,728` tag context for long documents without testing; publisher explicitly limits validated context to 8k.
- **English-Focused & Option-Order Sensitive**: Evaluations focus on English; add an explicit `none` option when choices may not cover the input.
- **No Base M4 Latency Measured**: Resident memory and JevBench were measured on M5 Max; base M4 timing for `kev:9b-nvfp4` is `TBD`.

## Sources

- `[2026-11-26]` [Ollama Library: kev](https://ollama.com/library/kev) & [kev/tags](https://ollama.com/library/kev/tags) — v0.40.4+ requirement, 1–255 option/score limits, M5 Max resident memory & JevBench v1.2.16 table, all 16 tags.
- `[2026-11-26]` [Hugging Face: jaredpalmer/kev-9b](https://huggingface.co/jaredpalmer/kev-9b/raw/main/README.md) & [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b/raw/main/README.md) — architecture, Apache-2.0 license, `transfer-v4` OOD/calibration results, 8,192-token validation limit.
