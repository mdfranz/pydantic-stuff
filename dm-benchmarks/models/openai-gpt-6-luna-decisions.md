# OpenAI `gpt-6-luna` Decisions (`openai-decisions:gpt-6-luna`)

## Metadata

| Field | Value |
| --- | --- |
| **Provider / Model ID** | `openai-decisions:gpt-6-luna` (Decisions API) vs. `openai:gpt-6-luna` (Responses API baseline) |
| **API Endpoint** | `POST https://api.openai.com/v1/decisions` (`OpenAIDecisionsModel`) `[Primary]` |
| **Architecture** | Dedicated closed-set multimodal Decision API backed by `gpt-6-luna` (zero output text tokens) `[Primary]` |
| **Context Window** | `~1.05M–1.1M` tokens (`gpt-6-luna` tier) `[Primary]` |
| **Pricing (USD)** | `$0.10 / 1M` input · `$0.01 / 1M` cached input · **`$0.00` output** on `openai-decisions` (vs. `$0.50 / 1M` output on `openai:gpt-6-luna`) `[Primary]` |
| **pydantic-ai Readiness** | `Out-of-the-box` (`OpenAIDecisionsModel`, `openai-decisions:gpt-6-luna`, `pydantic-ai-slim[openai]`) `[Measured]` |
| **Status** | `Shortlisted (Top Live-Verified Cloud Decision API)` |
| **Last Updated** | `2026-11-26` |

## API Limits, Pricing & Throughput Profile

- **Modalities Supported**: Text, conversation histories (including `function_call` / `function_call_output` items), structured JSON, and inline base64 or hosted image URLs `[Primary]`.
- **Hard Request Limits (verified in `pydantic_ai/models/openai_decisions.py` vs. OpenAI docs)**:
  - `max_questions`: `pydantic-ai 2.75.0` enforces **`200`** per request (`OpenAIDecisionsModel.max_questions = 200`, citing `openai-openapi`), whereas OpenAI's current `/v1/decisions` docs advertise up to **`400`** questions `[Primary]`.
  - `max_choice_options = 255` per `choice` question `[Primary]`.
  - `max_score_levels = 10` per `score` rubric `[Primary]`.
  - `max_images = 128` image parts across all input messages `[Primary]`.
  - `requires_instructions = False` (though empty instructions serialize as `""`).
- **Prompt Caching (`[Measured]`)**: OpenAI automatically cached `86,437 / 154,317` input tokens (`56.0%`) across the 231-item JevBench run with zero explicit cache markers, cutting total run cost to **`$0.00765`** (`~$0.000033` per decision).

## Decision-Model Assessment

- **Decision Quality & Determinism (`[Measured]`)**:
  - **JevBench Public (231 items)**: **`87.9%`** accuracy (`203/231`), **`82.2%`** chance-corrected accuracy, `100.0%` coverage (`0` errors).
  - **Tier Breakdown**: `Original` `94.4%`, `Easy` `100.0%`, `Hard` `78.4%`.
  - **Calibration**: **Brier `0.164`**, **ECE `0.026`**, **Ordinal MAE `0.18`** (`score` argmax `88.9%` / `pai-round` `88.9%`), **Paraphrase consistency `88.9%` (`88.9%` both ✓)**, **Trap rate `14.2%`** (`100.0%` on the `trap` family itself).
  - **Weak Spot (`[Measured]`)**: `temporal_numeric` hard family (`26.7%`) and `long_policy` (`73.7%`), though it achieved `100.0%` on `ambiguous`, `adversarial`, `routing_hard`, and `trap`.
- **Perturbation Robustness (`[Measured]`)**:
  - `order_reverse` flip: **`0.0%`** (`mean_abs_dp_gold 0.013`); `order_rotate` flip: **`0.0%`** (`0.007`); `anon_labels` flip: **`0.0%`**; `bundle` flip: **`0.0%`** (`0.000` Δp); `repeat` flip: **`0.0%`** (`0.000` Δp).
  - **Caveat — Gate / Negative Robustness (`[Measured]`)**: `none_option` accuracy was **`66.7%`** (`40/60`) and `polarity_swap` consistency was **`63.9%`** (`69.4%` acc) — notably softer than `claude-haiku-5-5` (`85.0%` / `88.9%`) and top local 2B System One models (`83.3%` / `97.2–100%`).
- **Option-Count Scaling (`[Measured]`)**:
  - `N=8`: `100.0% (12/12)` @ `160 ms` p50 · `N=32`: `100.0% (12/12)` @ `164 ms` · `N=64`: `100.0% (12/12)` @ `154 ms` · `N=128`: `100.0% (12/12)` @ `167 ms` · **`N=255`: `91.7% (11/12)` @ `192 ms` p50**.
- **Latency & `/v1/decisions` vs. `/v1/responses` (`[Measured]`)**:
  - On the 6-case / 8-question `smoke` suite: `openai-decisions:gpt-6-luna` scored **`8/8`** in **`252 ms`** warm median (`1,461` in / `0` out tokens), whereas generative `openai:gpt-6-luna` (`NativeOutput` via `/v1/responses`) scored **`8/8`** in **`868 ms`** warm median (**`3.4×` slower**, `723` in / `96` out tokens).
  - Across all 231 JevBench items (`concurrency=5`): **`164 ms` p50 / `512 ms` p95**.
- **Multimodal & `pydantic-ai` `ModelFallback` (`[Measured]`)**:
  - Red/Blue 16×16 PNG probe: **`2/2`** (`True` in `330.3 ms`, `False` in `130.8 ms`).
  - `ModelFallback` (`[FastTriage, EscalatedTriage]` -> `anthropic:claude-haiku-5-5`):
    - Routine ticket stayed on `gpt-6-luna` (`FastTriage`, **`235.0 ms`**, zero Anthropic call).
    - Legal escalation requiring free-form `rationale: str` triggered `UnfillableRoute` (`DecisionHandOff`) and cleanly handed off to `claude-haiku-5-5` (`EscalatedTriage`, `3,354.1 ms`).
    - Ambiguous gibberish (`"Hello, maybe billing or shipping or technical, not really sure yet."`) caused OpenAI's `/v1/decisions` to return **`"type": "refusal"`** on `FastTriage.team`, which `OpenAIDecisionsModel` raises as `ContentFilterError` — caught by `ModelFallback`'s default `fallback_on=(ModelAPIError, ContentFilterError, OutputTokenLimitExceeded)`, but bypassed if you narrow `fallback_on=DecisionHandOff`.

## Benchmarks & Evidence

- **Local Live Harness (`benchmarks/cloud_decisions_2026-11-26_123548_openai_anthropic_logfire.md`)** `[Measured]`: All metrics above verified end-to-end via `pydantic-ai-slim 2.75.0` with traces shipped to Logfire (`https://logfire-us.pydantic.dev/mdfranz/dm-testing`).
- **Artificial Analysis Decision Index** `[Benchmark]`: Published score `~60` (vs. `~66` for `gemini-decision-1` and `Jev 1.14`).

## Caveats & Open Questions

- `max_score_levels = 10` is tighter than Gemini Decision 1's `255` levels (sufficient for standard 0–4 / 0–9 rubrics).
- `none_option` (`66.7%`) and `polarity_swap` (`63.9%`) lag `claude-haiku-5-5`, so test negative/inverted gates explicitly before relying on zero-shot boolean flips.
- `pydantic-ai` caps `max_questions` at `200` (vs. `400` in OpenAI's newer docs) and enforces docstrings on union output models (`UserError` if omitted).
- Always use `openai-decisions:gpt-6-luna` (never `openai:gpt-6-luna`) when you want `/v1/decisions` probabilities and `3.4×` lower latency.

## Sources

- `[2026-11-26]` Installed source: `pydantic_ai/models/openai_decisions.py`, `providers/openai.py`, `genai-prices 0.2.5` — `[Primary]`
- `[2026-11-26]` OpenAI Decisions API (`POST /v1/decisions`) documentation — `[Primary]`
- `[2026-11-26]` Local benchmark (`benchmarks/cloud_decisions_2026-11-26_123548_openai_anthropic_logfire.md`) — `[Measured]`
