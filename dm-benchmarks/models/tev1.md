# Together AI `tev1` (`Tev1-0.8B` & `Tev1-4B` Experimental)

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `tev1:4b` / `latest` (`3ea3be2f2026`), `tev1:4b-nvfp4` (`1b15f25c13a0`), `tev1:4b-mxfp8` (`2a76e0606594`), `tev1:4b-q4_K_M` (`608e3a13eec3`), `tev1:4b-q8_0` (`7954dce22bc6`), `tev1:0.8b` (`cb16f3186b07`), `tev1:0.8b-mxfp8` (`c96262240cf8`), `tev1:0.8b-q8_0` (`abc31fdb0bf5`), plus `bf16` / `mlx-bf16` |
| **Parameters (Total / Active)** | `0.8B` / `0.8B` · `4B` / `4B` (dense) |
| **Architecture** | Supervised fine-tune of `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-4B` (retains standard next-token LM head; requires 2 token positions for scoring on Ollama) |
| **Context Window (Max / Practical M4)** | `256K` in registry tags / `8k–32k` practical M4 *(long-context eval not published)* |
| **Recommended Quantization** | `tev1:4b-nvfp4` (`3.1 GB`, MLX) or `tev1:4b-mxfp8` (`4.4 GB`, MLX); `tev1:0.8b-mxfp8` (`797 MB`, MLX) |
| **License** | **Unclear / Unfinalized for Fine-Tuned Weights** *(base Qwen3.5 is Apache-2.0, training scripts `github.com/togethercomputer/tev1` are MIT, but HF card states fine-tuned weights license is still being finalized)* |
| **pydantic-ai Readiness** | `Out-of-the-box` (`Minor config`: set `decision_max_choice_options=24`, `decision_max_score_levels=24`) |
| **Status** | `Candidate` (experimental; superseded by Apache-2.0, benchmarked `d3:4b`, `d1:2b`, `kev`, and `strands-decider`) |
| **Last Updated** | `2026-11-26` |

## Ollama Tags & Mac Mini M4 (24GB) Fit

Requires **Ollama v0.35.0 or later** `[Primary]`:

| Ollama Tag | Hash | Disk Size | Backend | M4 24GB Fit |
| --- | --- | ---: | --- | --- |
| `tev1:0.8b-mxfp8` | `c96262240cf8` | 797 MB | `MLX` | `Comfortable` |
| `tev1:0.8b-q8_0` | `abc31fdb0bf5` | 834 MB | GGUF | `Comfortable` |
| `tev1:0.8b-mlx-bf16` / `0.8b-bf16` | `9ea18bade940` / `e3e061c8b4e6` | 1.5 / 1.6 GB | `MLX` / GGUF | `Comfortable` |
| `tev1:4b-q4_K_M` | `608e3a13eec3` | 2.8 GB | GGUF | `Comfortable` |
| `tev1:4b-nvfp4` | `1b15f25c13a0` | 3.1 GB | `MLX` | `Comfortable` |
| `tev1:4b-mxfp8` (`4b`, `latest`) | `2a76e0606594` / `3ea3be2f2026` | 4.4 GB | `MLX` | `Comfortable` |
| `tev1:4b-q8_0` | `7954dce22bc6` | 4.6 GB | GGUF | `Comfortable` |
| `tev1:4b-mlx-bf16` / `4b-bf16` | `b6cc5c8eef21` / `4eb03a01d788` | 8.4 / 8.7 GB | `MLX` / GGUF | `Comfortable` |

- **Weight Size (Disk / Loaded)** `[Primary]`: `797 MB–1.6 GB` (`0.8B`), `2.8–8.7 GB` (`4B`).
- **Est. Total RAM @ 8k / 16k / 32k** `[Estimated]`: **~1.5–2.2 GB** (`0.8b-mxfp8`), **~4.0–4.8 GB** (`4b-nvfp4`), **~5.3–6.2 GB** (`4b-mxfp8`), **~9.5–10.5 GB** (`4b-mlx-bf16`) (`Comfortable` across all tags).
- **Fit Tier**: `Comfortable`.
- **Est. M4 Decode Speed** `[Estimated]`: `4b-nvfp4` (3.1 GB) ~70–180 ms est. single decision (~25–31 tok/s eq. bandwidth); `4b-mxfp8` (4.4 GB) ~95–220 ms est. (~18–22 tok/s eq.); `0.8b-mxfp8` (797 MB) ~25–65 ms est.

## Decision-Model Assessment

- **Decision Quality & Determinism** `[Primary]`: Experimental SFT of `Qwen3.5-0.8B` / `Qwen3.5-4B` trained to emit a single option letter (`A`–`X`, 2–24 options) without a specialized non-autoregressive pointer/decision head. Only internal bring-up dev results are published (`88.0%` on 1,000-row main dev set); lacks independent Decision Index or JevBench scores.
- **Structured Output & JSON Schema** `[Primary]`: Exposed through Ollama's `/v1/systemone` endpoint (`choice`, `noul`, `score`). Trained on **2–24 options** (`Ollama` notes staying within 2–24, and notes Tev1 requires two token positions for scoring).
- **Tool & Function Calling** `[Primary]`: Registry lists `['tools', 'decision', 'thinking']`, but Ollama documents it for `/v1/systemone` classification/routing rather than generative tool calling (`pydantic-ai` `SystemOneModel`).
- **Thinking / Latency Tradeoff** `[Primary]`: HF card configures `enable_thinking: false, max_tokens: 8, temperature: 0` (single-letter answer); `/v1/systemone` handles scoring via Ollama's runner.
- **Quantization & Context Notes** `[Primary]`: Calibration, prompt-injection resilience, multilingual behavior, and OOD robustness are explicitly listed as **not comprehensively evaluated** on the HF card.
- **Operational & pydantic-ai Readiness** `[Primary]`: **`Out-of-the-box`** (`Minor config`) via `SystemOneModel` (`'system-one:tev1:4b-mxfp8'` or `'system-one:tev1:0.8b-mxfp8'`) + `SystemOneProvider(base_url='http://localhost:11434')` + `DecisionModelProfile(decision_max_choice_options=24, decision_max_score_levels=24)`.

## Benchmarks & Evidence

- **Together AI Bring-up Dev Eval (`Tev1-4B-experimental`)** `[Primary]`:
  - Main decision dev set: `880 / 1,000` (`88.0%`).
  - Synthetic policy-transfer set: `300 / 300` (`100%`).
  - Valid single-letter outputs: `1,300 / 1,300`; HTTP errors: `0`.
  - *(Caveat: internal dev mixture with no untuned-Qwen baseline and no independent Decision Index / JevBench evaluation.)*

## Caveats & Open Questions

- **Unfinalized Weights License**: Fine-tuned weights license is still marked as "being finalized" on Hugging Face (`Tev1-4B-experimental`), and dataset sources lack a single blanket license—check before commercial use `[Primary]`.
- **Experimental / Unbenchmarked Externally**: No Decision Index, JevBench, or ECE calibration metrics published; `d3:4b`, `d1:2b`, `strands-decider:2b`, and `kev:9b-nvfp4` have stronger independent/public evidence and clear Apache-2.0 (or LFM) licensing.
- **2–24 Option Range**: Trained on 2–24 options; `0.8B` has no published accuracy figures on HF.

## Sources

- `[2026-11-26]` [Ollama Library: tev1](https://ollama.com/library/tev1) & [tev1/tags](https://ollama.com/library/tev1/tags) — 4B & 0.8B tags, 2–24 option guidance, `/v1/systemone` usage, v0.35.0+ requirement.
- `[2026-11-26]` [Hugging Face: togethercomputer/Tev1-4B-experimental](https://huggingface.co/togethercomputer/Tev1-4B-experimental/raw/main/README.md) & [Tev1-0.8B-experimental](https://huggingface.co/togethercomputer/Tev1-0.8B-experimental/raw/main/README.md) — `Qwen3.5` SFT architecture, bring-up dev results (`88.0%`), limitations, unfinalized weights license.
