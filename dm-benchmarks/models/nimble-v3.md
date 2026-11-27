# Bespoke Labs Nimble v3 (`nimble-v3`)

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `nimble-v3:9b-nvfp4` (`a11fb440f1e6`), `nimble-v3:9b-mxfp8` / `9b` / `latest` (`d14c05dc038c` / `4f30b1db22d3`), `nimble-v3:9b-q4_K_M` (`88cbbd22b816`), `nimble-v3:9b-q8_0` (`6f0e87f39800`), `nimble-v3:9b-mlx-bf16` (`11dce4c7532f`), `nimble-v3:9b-bf16` (`1873ffdc0e99`) |
| **Parameters (Total / Active)** | `9B` / `9B` (dense) |
| **Architecture** | `Qwen/Qwen3.5-9B` + LoRA decision adapter (`32` layers: 24 Gated DeltaNet + 8 full attention) |
| **Context Window (Max / Practical M4)** | `32K` (`32,768` tokens) / `32K` practical M4 |
| **Recommended Quantization** | `nimble-v3:9b-nvfp4` (`6.9 GB`, MLX) or `nimble-v3:9b-mxfp8` (`9.3 GB`, MLX) |
| **License** | **CC BY-NC 4.0 (Noncommercial Use Only)** *(base `Qwen/Qwen3.5-9B` is Apache-2.0; Nimble v3 weights prohibit commercial use)* |
| **pydantic-ai Readiness** | `Out-of-the-box` (`SystemOneModel` via `/v1/systemone`) |
| **Status** | `Shortlisted (Noncommercial Only)` — top 9B JevBench accuracy on 24GB M4, excluded for commercial projects |
| **Last Updated** | `2026-11-26` |

## Ollama Tags & Mac Mini M4 (24GB) Fit

Requires **Ollama v0.40.4 or later** `[Primary]`. JevBench 231 accuracy & HTTP latency below were measured on an Apple M5 Max (40-core GPU, Ollama `0.40.4-rc0`, prompts 190–4,112 tokens) `[Primary]`:

| Ollama Tag | Format / Hash | Disk Size | JevBench (231) `[Primary]` | M5 Max Median / p95 `[Primary]` | M4 24GB Fit |
| --- | --- | ---: | ---: | ---: | --- |
| `nimble-v3:9b-q4_K_M` | GGUF (`88cbbd22b816`) | 5.6 GB | 198 / 231 | 168 ms / 1,344 ms | `Comfortable` |
| `nimble-v3:9b-nvfp4` | MLX (`a11fb440f1e6`) | **6.9 GB** | **201 / 231** | **99 ms / 929 ms** | `Comfortable` |
| `nimble-v3:9b-mxfp8` (`9b`, `latest`) | MLX (`d14c05dc038c` / `4f30b1db22d3`) | 9.3 GB | **201 / 231** | 103 ms / 1,057 ms | `Comfortable` |
| `nimble-v3:9b-q8_0` | GGUF (`6f0e87f39800`) | 9.5 GB | 200 / 231 | 163 ms / 1,273 ms | `Comfortable` |
| `nimble-v3:9b-mlx-bf16` | MLX (`11dce4c7532f`) | 18 GB | 201 / 231 | 88 ms / 873 ms | `Too Large` |
| `nimble-v3:9b-bf16` | GGUF (`1873ffdc0e99`) | 18 GB | 202 / 231 | 176 ms / 1,336 ms | `Too Large` |

- **Weight Size (Disk / Loaded)** `[Primary]`: `6.9 GB` (`9b-nvfp4`), `9.3 GB` (`9b-mxfp8`).
- **Est. Total RAM @ 8k / 16k / 32k** `[Estimated]` (`Qwen3.5-9B` KV cache ~32 KB/tok + ~1.0–1.5 GB Metal overhead):
  - `nimble-v3:9b-nvfp4` (6.9 GB): **~8.2–9.0 GB** @ 8k–32k (`Comfortable`).
  - `nimble-v3:9b-mxfp8` (9.3 GB): **~10.6–11.5 GB** @ 8k–32k (`Comfortable`).
- **Fit Tier**: `Comfortable` (`nvfp4`, `mxfp8`, `q4_K_M`, `q8_0`); `Too Large` (`bf16` 18 GB tags).
- **Est. M4 Decode Speed** `[Estimated]`: Single-pass decision (no reasoning trace generated). Base M4 (~120 GB/s) has roughly ~22% of M5 Max bandwidth, so expect ~200–450 ms median on `9b-nvfp4` (~11.3–13.9 tok/s eq. decode bandwidth) and ~250–500 ms on `9b-mxfp8` (~8.4–10.3 tok/s eq.). Note that MLX runs ~1.6–1.7× faster than GGUF on Apple Silicon in Ollama's M5 Max table (`99 ms` vs `163–168 ms`) `[Primary]`.

## Decision-Model Assessment

- **Decision Quality & Determinism** `[Primary]`: Strongest 9B text decision accuracy tested on JevBench (`201/231` on `9b-nvfp4`/`mxfp8`, only 1 below `bf16` GGUF `202/231`) and `56.88` on Decision Index 0.2.1. Notably, `NVFP4` matches `MXFP8` and `MLX BF16` at `201/231` with `2.4 GB` less disk/RAM.
- **Structured Output & JSON Schema** `[Primary]`: Uses `/v1/systemone` (`choice`, `noul`, `score`) returning calibrated probabilities without generating prose.
- **Tool & Function Calling** `[Primary]`: Although the Ollama registry lists `['tools', 'decision', 'thinking']` capability tags, the model page explicitly documents `/v1/systemone` direct token scoring ("without generating a reasoning trace or a free-form response"). Use via `SystemOneModel` + `ModelFallback` for tool execution.
- **Thinking / Latency Tradeoff** `[Primary]`: Scores candidate tokens directly without generating a reasoning trace.
- **Quantization & Context Notes** `[Primary]`: `32,768`-token context; MLX variants (`9b-nvfp4`, `9b-mxfp8`) strictly beat GGUF (`q4_K_M`, `q8_0`) on both accuracy and Apple Silicon latency in Ollama's tests.
- **Operational & pydantic-ai Readiness** `[Primary]`: **`Out-of-the-box`** via `SystemOneModel` (`'system-one:nimble-v3:9b-nvfp4'`) + `SystemOneProvider(base_url='http://localhost:11434')` + `DecisionModelProfile(decision_max_choice_options=26, decision_max_score_levels=26, context_window=32768)`.

## Benchmarks & Evidence

- **Decision Index 0.2.1 (Bespoke Labs reference run)** `[Primary]`: **`56.88`**.
- **JevBench (231 unmodified cases, prompts 190–4,112 tokens, Apple M5 Max + Ollama `0.40.4-rc0`)** `[Primary]`:
  - `nimble-v3:9b-nvfp4` (MLX): **`201/231`**, `99 ms` median / `929 ms` p95.
  - `nimble-v3:9b-mxfp8` (MLX): **`201/231`**, `103 ms` median / `1,057 ms` p95.
  - `nimble-v3:9b-q8_0` (GGUF): `200/231`, `163 ms` median / `1,273 ms` p95.
  - `nimble-v3:9b-q4_K_M` (GGUF): `198/231`, `168 ms` median / `1,344 ms` p95.

## Caveats & Open Questions

- **Noncommercial License Only (CC BY-NC 4.0)**: Unlike the earlier `nimble` release or Apache-2.0 `d3`/`kev`/`strands-decider`, `nimble-v3` **cannot** be used commercially without separate licensing from Bespoke Labs `[Primary]`.
- **Text-Only**: No image/vision support.
- **M5 Max vs. Base M4**: Latency numbers above are from a 40-core M5 Max; base M4 latency will be higher (`TBD`).

## Sources

- `[2026-11-26]` [Ollama Library: nimble-v3](https://ollama.com/library/nimble-v3) & [nimble-v3/tags](https://ollama.com/library/nimble-v3/tags) — CC BY-NC 4.0 notice, v0.40.4+ requirement, 32K ctx, M5 Max JevBench 231 benchmark table, all 8 tags.
- `[2026-11-26]` [Hugging Face: bespokelabs/Bespoke-Nimble-9B-v3](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3/raw/main/README.md) — `Qwen/Qwen3.5-9B` LoRA details, Decision Index 0.2.1 (`56.88`), CC BY-NC 4.0 license.
