# AutoTrust JEV (`autotrust-jev`)

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `autotrust-jev:9b-nvfp4` (`454a99f93908`), `autotrust-jev:9b` / `autotrust-jev:latest` / `autotrust-jev:9b-mxfp8` (`ae39b218bea7`) *(plus `9b-mlx-bf16`, `27b-vl*`, `35b*` — Too Large for 24GB M4)* |
| **Parameters (Total / Active)** | 9B variant: ~7.9B text backbone + 40.2M adapter (9B class) / ~9B (`Qwen/Qwen3.5-9B`); 35B MoE: 35B / ~3B active |
| **Architecture** | Dense hybrid (`qwen3_5_text`: 24 Gated DeltaNet `linear_attention` + 8 `full_attention` layers) packaged on Ollama as `AutoTrustDecision` (`MLX`) |
| **Context Window (Max / Practical M4)** | 256K (`262,144`) max / 8k–32k practical M4 *(note: System 1 corpus states averaged ≤856 tokens; local `/v1/systemone` caps text+schema at 64 KiB)* |
| **Recommended Quantization** | `NVFP4` (`autotrust-jev:9b-nvfp4`, 8.0 GB) for daily headroom; `MXFP8` (`autotrust-jev:9b`, 11 GB) for higher precision |
| **License** | Apache-2.0 (`Qwen/Qwen3.5-9B` and `SargeDev/jev-distill-corpus-v3` are Apache-2.0) |
| **pydantic-ai Readiness** | `Out-of-the-box` (`Minor config` recommended for `DecisionModelProfile` caps) |
| **Status** | `Shortlisted` (`9b-nvfp4` and `9b`/`9b-mxfp8`); larger tags `Excluded` (`Too Large`) |
| **Last Updated** | `2026-11-26` |

## Ollama Tags & Mac Mini M4 (24GB) Fit

All official `autotrust-jev` tags on Ollama use the `MLX` runner `[Primary]`:

| Ollama Tag | Hash | Disk Size | Input | M4 24GB Fit | Notes |
| --- | --- | --- | --- | --- | --- |
| `autotrust-jev:9b-nvfp4` | `454a99f93908` | 8.0 GB | Text | `Comfortable` | Best M4 24GB daily driver; NVFP4 MLX quant |
| `autotrust-jev:9b` / `latest` / `9b-mxfp8` | `ae39b218bea7` | 11 GB | Text | `Comfortable` / `Tight` | Near 12 GB sweet spot; tight if paired with long ctx or heavy OS load |
| `autotrust-jev:9b-mlx-bf16` | `18e60050218c` | 18 GB | Text | `Too Large` | Exceeds ~16–18 GB usable M4 RAM ceiling |
| `autotrust-jev:27b-vl` / `27b-vl-nvfp4` | `ddd4abaa2baf` | 21 GB | Text, Image | `Too Large` | `Qwen/Qwen3.8-27B` backbone; exceeds 24GB budget |
| `autotrust-jev:27b-vl-mxfp8` | `3006be8ff2e9` | 32 GB | Text, Image | `Too Large` | Exceeds 24GB total unified memory |
| `autotrust-jev:35b` / `35b-nvfp4` | `973a7b9bdf9f` | 26 GB | Text, Image | `Too Large` | `Qwen/Qwen3.5-35B-A3B` MoE; total weights exceed 24GB |
| `autotrust-jev:35b-mxfp8` | `4a51b82c2ffa` | 39 GB | Text, Image | `Too Large` | Exceeds 24GB total unified memory |

- **Weight Size (Disk / Loaded)** `[Primary]`: `8.0 GB` (`9b-nvfp4`) or `11 GB` (`9b` / `9b-mxfp8`).
- **KV Cache Math** `[Estimated]`: Only 8 of 32 layers use `full_attention` (`num_key_value_heads=4`, `head_dim=256`). At fp16, KV cache is `8 × 2 × 4 × 256 × 2 B = 32 KB/token` (~256 MB @ 8k, ~512 MB @ 16k, ~1.0 GB @ 32k), plus small fixed recurrent state for the 24 `linear_attention` layers.
- **Est. Total RAM @ 8k / 16k / 32k** `[Estimated]` (weights + KV cache + ~1.0–1.5 GB Metal/MLX runtime overhead):
  - `autotrust-jev:9b-nvfp4` (8.0 GB): **~9.3 GB** @ 8k / **~9.5 GB** @ 16k / **~10.0 GB** @ 32k (`Comfortable`).
  - `autotrust-jev:9b` / `9b-mxfp8` (11 GB): **~12.3 GB** @ 8k / **~12.5 GB** @ 16k / **~13.0 GB** @ 32k (`Comfortable` upper edge at 8k–16k, `Tight` under concurrent workload).
- **Fit Tier**: `Comfortable` (`9b-nvfp4`), `Comfortable`–`Tight` (`9b` / `9b-mxfp8`).
- **Est. M4 Decode Speed** `[Estimated]`:
  - System One decisions perform **one prefill pass (0–1 output tokens)**, not multi-token decoding.
  - Weight-streaming lower bound at M4 120 GB/s: ~67 ms (`9b-nvfp4`) to ~92 ms (`9b-mxfp8`), plus prompt-eval compute (~150–400 ms estimated total per decision for typical 200–500 token prompts; unmeasured on M4).
  - Equivalent single-token decode bandwidth sanity check (`120 GB/s ÷ size_GB × 0.65–0.80`): ~9.8–12.0 tok/s (`9b-nvfp4`) and ~7.1–8.7 tok/s (`9b-mxfp8`).
- **Recommended Ollama Env / Settings**: Requires **Ollama v0.35.0+** `[Primary]`. Uses Ollama's native `MLX` runner (`NVFP4`/`MXFP8`); `Gated DeltaNet` hybrid layers already keep KV cache very small.

## Decision-Model Assessment

- **Decision Quality & Determinism** `[Primary]` `[Benchmark]`: Purpose-built System One decision model distilled from TypeSafe Jev 1.13 (`noul` yes/no, `choice` multi-option, `score` 0–5 rubric). Strong binary calibration (`0.996` AUROC, `0.0007` ECE on `test_set_30k`), `91.8%` OOD top-1 accuracy, and `95.4%` top-1 agreement on decisive `choice` rows. Independent of and unaffiliated with TypeSafe AI.
- **Structured Output & JSON Schema** `[Primary]`: Does **not** do free-form JSON generation or open-ended `str` extraction. Instead, `/v1/systemone` directly returns typed JSON `answers` (`choice`, `noul` probability `0–1`, `score` distribution + confidence) in one pass, avoiding JSON syntax errors and Pydantic validation retries on supported field types.
- **Tool & Function Calling** `[Primary]`: **Not supported** via Ollama's `/v1/systemone` API (`tools` are explicitly unsupported on `/v1/systemone`). However, `pydantic-ai`'s `DecisionModel` layer can use `autotrust-jev` as a **router** over tool definitions/output functions and escalate via `ModelFallback` to a generative LLM when a chosen tool needs free-form arguments.
- **Thinking / Latency Tradeoff** `[Primary]`: **Single-pass System 1 only on Ollama** (0–1 output tokens, no CoT thinking overhead). Note: On Hugging Face/vLLM, `autotrust/JEV-9B` includes a separate System 2 path (unmodified `Qwen3.5-9B` `lm_head`), whereas Ollama's `autotrust-jev` manifest uses `"architectures": ["AutoTrustDecision"]` for `/v1/systemone` only.
- **Quantization & Context Notes** `[Primary]` `[Estimated]`: All Ollama 9B tags are MLX quantizations (`NVFP4`, `MXFP8`, or `BF16`); Hugging Face benchmarks were measured on `bf16` weights (`17.9 GB`), so local calibration/probability drift under `9b-nvfp4` vs. `9b-mxfp8` has not yet been measured on M4 (though on `JEV-35B`, AutoTrust observed NVFP4 retained `96.6%` identical answers to `bf16` on Decision Index). Ollama's 9B tags are **Text-only** (vision requires `27b-vl` or `35b`, which are too large for 24GB M4).
- **Operational & pydantic-ai Readiness** `[Primary]`: **`Out-of-the-box`** (with **`Minor config`** recommended).
  - **Provider & Endpoint**: Use `pydantic-ai`'s `SystemOneModel` (`'system-one:autotrust-jev:9b-nvfp4'` or `'system-one:autotrust-jev:9b'`) with `SYSTEM_ONE_BASE_URL='http://localhost:11434'` (or `SystemOneProvider(base_url='http://localhost:11434')`) — **do NOT append `/v1`** (unlike Ollama's OpenAI-compatible chat provider).
  - **Supported Pydantic types**: `bool` (`noul`), `Literal`/`Enum` (`choice`), `list[Literal]` (multi-label `noul`), bounded `float` (`noul` probability), `IntEnum` (`score`), and nested `BaseModel`s of these types. Free-form `str` or unbounded `int`/`float` raises `UserError` unless routed via `ModelFallback`.
  - **Recommended `DecisionModelProfile`**: Set `DecisionModelProfile(decision_max_choice_options=16, decision_max_score_levels=6)` to match JEV-9B's 24-slot decision head (`noul` 2, `score` 6 levels `0–5`, `choice` 2–16 options). Ollama's generic `/v1/systemone` server cap is 26 options / 26 levels, so verify locally if you pass >16 choices or >6 score levels.
  - **Minimal `pydantic-ai` wiring**:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai.models.system_one import SystemOneModel
    from pydantic_ai.profiles.decision import DecisionModelProfile
    from pydantic_ai.providers.system_one import SystemOneProvider

    model = SystemOneModel(
        'autotrust-jev:9b-nvfp4',
        provider=SystemOneProvider(base_url='http://localhost:11434'),
        profile=DecisionModelProfile(
            decision_max_choice_options=16,
            decision_max_score_levels=6,
        ),
    )
    ```

## Benchmarks & Evidence

- **Held-out `test_set_30k` (`SargeDev/jev-distill-corpus-v3`, `bf16` checkpoint)** `[Primary]`:
  - Mean KL divergence to TypeSafe Jev 1.13 (25,376 Jev-labelled rows): `≈ 0.019` (`0.021` across all 29,955 rows).
  - `noul` AUROC: `0.996` (all rows) / `0.994` (Jev-labelled); `noul` Brier: `0.0015`.
  - `score` MAE (0–5 scale): `0.103`; Calibration ECE (15 bins): `0.0007`.
  - `choice` top-1 agreement with Jev: `90.2%` (Jev-labelled) / `89.8%` (all rows) / `95.4%` (decisive rows, top-2 gap ≥ 0.1).
  - Unseen task families (`openjev_v2` OOD split): KL `0.234`, top-1 accuracy `91.8%`.
  - Option-shuffle top-1 flip rate (test set): `3.9%`.
- **Independent `gazelle93/decision-models-under-pressure` (800 items, human gold labels, AutoTrust re-run up to 16 options)** `[Benchmark]`:
  - Accuracy @ 2 / 4 / 8 / 16 options: `86.8% / 77.4% / 73.5% / 69.4%` (`97% / 97% / 94% / 90%` of TypeSafe Jev 1.13's `89.0% / 80.1% / 78.2% / 76.9%`).
  - 16-option near-miss vs. unrelated wrong options (CLINC + MTOP, 400 items): `87.5%` vs. `97.5%`.
  - 16-option answers changed by option order alone: `11.5%` (vs. Jev 1.13 `7.0%`, JEV-27B `7.4%`).
- **System 2 Reference (HF base `Qwen3.5-9B`, not exposed on Ollama `/v1/systemone`)** `[Primary]`: HumanEval pass@1 `70.7%`.
- **Reference Throughput (1× NVIDIA B200, `bf16`)** `[Primary]`: `≈ 90 ms` median single-request latency (`87 ms`), `2.5 ms`/decision batched (128 batch), `≈ 340` decisions/s. *(Caveat: datacenter GPU numbers; Mac Mini M4 latency is `TBD`.)*

## Caveats & Open Questions

- **No M4 Measurements Yet**: Actual single-pass latency (ms) and probability calibration drift (`9b-nvfp4` vs. `9b-mxfp8` vs. HF `bf16`) have not yet been measured locally on the Mac Mini M4.
- **No Native Text Generation / Tool Execution on Ollama**: Cannot generate free-form `str` fields, summaries, or complex tool arguments; pair with a System 2 generative Ollama model via `pydantic-ai`'s `ModelFallback` if needed.
- **Option-Count & Ordering Sensitivity**: Accuracy drops from `97%` of teacher (2–4 options) to `90%` of teacher (16 options), with `11.5%` order sensitivity at 16 options. AutoTrust notes JEV-9B is strongest for ≤ ~8 options, routing, triage, and moderation.
- **Context & Language Scope**: Despite the 256K context window in config, corpus training states averaged ≤ 856 tokens (HF `serve.sh` truncates states >1,024 tokens; Ollama `/v1/systemone` enforces a 64 KiB text+schema cap without truncation). Training corpus is English-centric; `yuri_v1` memory-relevance rows in the corpus are `[0.5, 0.5]` placeholders (do not use zero-shot for memory relevance without testing).
- **Not Evaluated on Decision Index 0.3 / JevBench**: Unlike `JEV-27B-VL` (`53.64`) and `JEV-35B` (`59.73`), `JEV-9B` does not have a published Decision Index 0.3 or JevBench score.

## Sources

- `[2026-11-26]` [Ollama Library: autotrust-jev](https://ollama.com/library/autotrust-jev) — model overview, CLI usage, headline metrics.
- `[2026-11-26]` [Ollama Tags: autotrust-jev/tags](https://ollama.com/library/autotrust-jev/tags) — exact tags, sizes, MLX runner, hashes, context limits.
- `[2026-11-26]` [Ollama Docs: Decision Capabilities](https://docs.ollama.com/capabilities/decision.md) & [System One API](https://docs.ollama.com/api/systemone.md) — `/v1/systemone` spec, limits, v0.35.0+ requirement.
- `[2026-11-26]` [Hugging Face: autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B/raw/7ac669b8fb476ba926491e3d1c39f9330d68dbed/README.md) & [config.json](https://huggingface.co/autotrust/JEV-9B/raw/7ac669b8fb476ba926491e3d1c39f9330d68dbed/config.json) — architecture, Blocks of Experts recipe, `test_set_30k` and independent benchmarks, caveats.
- `[2026-11-26]` [Hugging Face: autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL/raw/9e53b8a809d2ab98e3e6a55b2b2dbf94b4c1e84a/README.md) & [autotrust/JEV-35B](https://huggingface.co/autotrust/JEV-35B/raw/a5d0d580bb9324f0f4c07dcba5c5ca91e813e7c9/README.md) — sibling model context and NVFP4 retention data.
- `[2026-11-26]` [Pydantic AI Docs: System One API](https://pydantic.dev/docs/ai/models/system-one/index.md) & [Decision Models](https://pydantic.dev/docs/ai/models/decision/index.md) — `SystemOneModel`, `DecisionModelProfile`, supported types, and `ModelFallback` routing.
