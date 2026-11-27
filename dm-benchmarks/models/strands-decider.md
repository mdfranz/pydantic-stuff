# Strands Decider (`strands-decider`)

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `strands-decider:2b-mxfp8` / `2b` / `latest` (`fd88a2b3ab3f`), `strands-decider:2b-nvfp4` (`c92a6f466917`), `strands-decider:2b-mlx-bf16` (`d7afa63135f6`) |
| **Parameters (Total / Active)** | `2B` / `2B` (dense) |
| **Architecture** | `Qwen/Qwen3.5-2B-Base` + 3-seed LoRA soup + trained readout head (`v1-2610`), `MLX`-only on Apple Silicon |
| **Context Window (Max / Practical M4)** | `4K` (`4,096` tokens max in Ollama tags) / `4K` practical M4 |
| **Recommended Quantization** | `strands-decider:2b-mxfp8` (`2.0 GB`, MLX, default) or `strands-decider:2b-nvfp4` (`1.3 GB`, MLX) |
| **License** | Apache-2.0 |
| **pydantic-ai Readiness** | `Out-of-the-box` (`Minor config`: set `context_window=4096`) |
| **Status** | `Shortlisted` (compact, Apache-2.0 2B text classifier with strong JevBench) |
| **Last Updated** | `2026-11-26` |

## Ollama Tags & Mac Mini M4 (24GB) Fit

Requires **Ollama v0.40.0 or later** on Apple Silicon (`MLX`-only) `[Primary]`. JevBench 231 scores and timings below were measured on an Apple M5 Pro (64 GiB RAM, macOS 27.0, Ollama `0.40.0`, prompts 82–3,936 tokens) `[Primary]`:

| Ollama Tag | Format / Hash | Disk Size | JevBench (231) `[Primary]` | M5 Pro Median / p95 `[Primary]` | M4 24GB Fit |
| --- | --- | ---: | ---: | ---: | --- |
| `strands-decider:2b-nvfp4` | MLX NVFP4 + MXFP8 embeddings (`c92a6f466917`) | 1.3 GB | **182 / 231** | **41.5 ms / 372.8 ms** | `Comfortable` |
| `strands-decider:2b-mxfp8` (`2b`, `latest`) | MLX MXFP8 (`fd88a2b3ab3f`) | 2.0 GB | **183 / 231** | 42.5 ms / 384.8 ms | `Comfortable` |
| `strands-decider:2b-mlx-bf16` | MLX BF16 (`d7afa63135f6`) | 3.8 GB | **182 / 231** | 33.3 ms / 343.5 ms | `Comfortable` |

- **Weight Size (Disk / Loaded)** `[Primary]`: `1.3 GB` (`nvfp4`), `2.0 GB` (`mxfp8`), `3.8 GB` (`mlx-bf16`).
- **Est. Total RAM @ 4k (max context)** `[Estimated]`: **~2.0–2.8 GB** (`2b-nvfp4` / `2b-mxfp8`) to **~4.5 GB** (`2b-mlx-bf16`) (`Comfortable`).
- **Fit Tier**: `Comfortable` across all tags (minimal footprint; easily stays co-resident with larger System 2 models).
- **Est. M4 Decode Speed** `[Estimated]`: Single forward pass; M4 120 GB/s weight-read floor for `2b-mxfp8` (2.0 GB) is ~17 ms (`nvfp4` 1.3 GB ~11 ms), so expect ~55–140 ms per text decision on base M4 (~39–48 tok/s eq. decode bandwidth on `mxfp8`, ~60–74 tok/s eq. on `nvfp4`).

## Decision-Model Assessment

- **Decision Quality & Determinism** `[Primary]`: Delivers `183/231` on JevBench at just 2B (`2b-mxfp8`), matching `kev:9b-nvfp4` (`183/231`) on that benchmark while using ~1/5 the memory. However, on the broader **Decision Index (balanced skill)**, it scores **`30.82`** (lower than `d3:2b`'s `38.9` and `d1-2B`'s `51.1` public suite), indicating narrower generalization outside JevBench-style tasks.
- **Structured Output & JSON Schema** `[Primary]`: Native `/v1/systemone` endpoint (`choice`, `noul`, `score`). Hugging Face eval reports `1.000` strict schema compliance on JevBench.
- **Tool & Function Calling** `[Primary]`: `/v1/systemone` router only (use `ModelFallback` in `pydantic-ai` for tool execution/free-form parameters).
- **Thinking / Latency Tradeoff** `[Primary]`: Single forward pass; zero output tokens.
- **Quantization & Context Notes** `[Primary]`: Hard **4,096-token context** cap on Ollama. Decision head stays at original precision in `nvfp4` and `mxfp8` builds (`NVFP4` `182/231` vs `MXFP8` `183/231` vs `BF16` `182/231` — virtually zero quantization degradation).
- **Operational & pydantic-ai Readiness** `[Primary]`: **`Out-of-the-box`** via `SystemOneModel` (`'system-one:strands-decider:2b-mxfp8'`) + `SystemOneProvider(base_url='http://localhost:11434')` + `DecisionModelProfile(decision_max_choice_options=26, decision_max_score_levels=10, context_window=4096)`.

## Benchmarks & Evidence

- **Ollama JevBench (231 cases, M5 Pro, Ollama `0.40.0`)** `[Primary]`: `2b-mxfp8` **`183/231`** (`42.5 ms` median), `2b-nvfp4` `182/231` (`41.5 ms` median), `2b-mlx-bf16` `182/231` (`33.3 ms` median).
- **HF Reference (`StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`, 4096 window on L4)** `[Primary]`:
  - JevBench public: `180/231` (`48` easy / `67` standard / `65` hard), Brier `0.280`, ECE `0.072`, strict schema `1.000`.
  - Decision Index (balanced skill): `30.82`.

## Caveats & Open Questions

- **Text-Only on Ollama**: HF repo mentions optional `--vision`, but Ollama `strands-decider` tags accept **Text only**.
- **4K Context Ceiling**: Prompts + questions must fit within 4,096 tokens.
- **Divergence Between JevBench & Decision Index**: Very strong JevBench score (`183/231`) vs. modest Decision Index (`30.82`). Publisher also notes long multi-step documents are a weak spot and question wording is weighted less than document text; calibration uses one temperature per primitive fitted on short held-out examples.
- **M5 Pro vs. Base M4**: Published Ollama timings are from M5 Pro; base M4 latency is `TBD`.

## Sources

- `[2026-11-26]` [Ollama Library: strands-decider](https://ollama.com/library/strands-decider) & [strands-decider/tags](https://ollama.com/library/strands-decider/tags) — v0.40.0+ MLX requirement, 4K context, Apple M5 Pro JevBench 231 timings, tags.
- `[2026-11-26]` [Hugging Face: StrandsAgents/strands-decider-2B-qwen3.5-v1-2610](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/raw/main/README.md) — `Qwen3.5-2B-Base` 3-seed soup architecture, Apache-2.0 license, Decision Index (`30.82`), Brier/ECE, limitations.
