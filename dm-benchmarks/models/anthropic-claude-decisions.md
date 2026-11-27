# Anthropic Claude 5.5 for Decision Workloads (`claude-haiku-5-5` & `claude-sonnet-5-5`)

## Metadata

| Field | Value |
| --- | --- |
| **Provider / Model IDs** | `anthropic:claude-haiku-5-5` (primary fast tier) · `anthropic:claude-sonnet-5-5` (System-2 / fallback tier) |
| **API Endpoint** | `POST https://api.anthropic.com/v1/messages` (`AnthropicModel`) — **No dedicated `/v1/decisions` endpoint** `[Primary]` |
| **Architecture** | Generative autoregressive LLMs with grammar-constrained `NativeOutput` (`output_config.format` JSON schema) & Adaptive Thinking `[Primary]` |
| **Pricing (USD)** | `claude-haiku-5-5`: `$0.10 / 1M` in (≤100K), `$0.50 / 1M` out, `$0.01 / 1M` cache read · `claude-sonnet-5-5`: `$2.00 / 1M` in, `$10.00 / 1M` out, `$0.10 / 1M` cache read `[Primary]` |
| **pydantic-ai Readiness** | `Out-of-the-box` (`NativeOutput` + `ModelSettings(thinking=False)`, `pydantic-ai-slim[anthropic]`) `[Measured]` |
| **Status** | `Shortlisted (Top Accuracy / Robustness & Best ModelFallback Target)` |
| **Last Updated** | `2026-11-26` |

## API Limits, Pricing & Operational Profile

- **No Native Decision / System One API (`[Primary]`)**: As of `2026-11-26`, Anthropic does **not** offer a zero-output-token `/v1/decisions` or `/v1/systemone` endpoint, nor does `pydantic-ai` have an `AnthropicDecisionsModel`. Decision tasks run through `POST /v1/messages` using `NativeOutput(Schema)` (`output_config.format` constrained `json_schema` decoding) or strict tool use (`tools[].strict: true`).
- **Adaptive Thinking & Sampling Quirks (`[Primary]` & `[Measured]`)**:
  - Both `claude-haiku-5-5` (`medium` default) and `claude-sonnet-5-5` (`high` default) use Adaptive Thinking. Legacy `thinking: {"type": "enabled", "budget_tokens": N}` is rejected with HTTP 400 `[Primary]`.
  - **Sampling parameters unsupported (`[Measured]`)**: Passing `temperature=0.0` triggers `UserWarning: Sampling parameters ['temperature'] are not supported by 'claude-haiku-5-5'. These settings will be ignored.` in `pydantic-ai 2.75.0` (same for `claude-sonnet-5-5`). Omit `temperature` and pass `ModelSettings(thinking=False)` cleanly.
- **Prompt Caching (`[Measured]`)**: Unlike OpenAI's automatic prefix caching (`56%` hit rate on JevBench), Anthropic recorded **`0` `cache_read_tokens`** without explicit `CachePoint` / `anthropic_cache_tool_definitions` markers (`[Measured]`), increasing effective token cost on repeated long policies unless explicit cache breakpoints are added.

## Decision-Model Assessment (`claude-haiku-5-5` Primary; `claude-sonnet-5-5` Reference)

- **Decision Quality & Determinism (`[Measured]`)**:
  - **Smoke Suite (6 cases / 8 questions)**: Both `claude-haiku-5-5` (`1,014 ms` warm med, `$0.000247`) and `claude-sonnet-5-5` (`911 ms` warm med, `$0.004916`, `20×` costlier) scored **`8/8`**.
  - **JevBench Public (231 items, `claude-haiku-5-5`)**: **`89.2%`** accuracy (`206/231`), **`84.1%`** chance-corrected accuracy, `100.0%` coverage (`0` errors) — **beating `openai-decisions:gpt-6-luna` (`87.9%`)**.
  - **Tier Breakdown**: `Original` **`97.2%`**, `Easy` **`100.0%`**, `Hard` **`79.3%`**.
  - **Ordinal & Consistency**: **Ordinal MAE `0.11`** (`score` accuracy `88.9%`), **Paraphrase consistency `94.4%` (`94.4%` both ✓)**, **Trap rate `14.2%`** (`100.0%` on `trap` family).
  - **Hard-Tier Strengths vs. `gpt-6-luna` (`[Measured]`)**: Higher on `temporal_numeric` (`46.7%` vs. `26.7%`), `long_policy` (`78.9%` vs. `73.7%`), and `multi_hop` (`88.9%` vs. `83.3%`); lower on `ambiguous` (`71.4%` vs. `100.0%`) and `tradeoff` (`66.7%` vs. `83.3%`).
- **Calibration Gap (`[Measured]`)**: Because `POST /v1/messages` returns discrete JSON values rather than calibrated label probabilities, **Brier and ECE are unavailable (`—`)**, and `pydantic-ai`'s `decision_boolean_threshold` / `decision_route_threshold` probability gating cannot be used natively.
- **Perturbation Robustness (`[Measured]`)**:
  - `order_reverse` flip: **`0.0%`**; `order_rotate` flip: **`0.0%`**; `repeat` flip: **`0.0%`**; `anon_labels` flip: **`1.4%`** (`1/72`).
  - **Standout Robustness vs. `gpt-6-luna` (`[Measured]`)**: `none_option` accuracy **`85.0%`** (`51/60` vs. `66.7%`) and `polarity_swap` consistency **`88.9%`** (`94.4%` acc vs. `63.9%` / `69.4%`).
- **Latency & Cost Tradeoff (`[Measured]`)**:
  - **JevBench (231 items)**: **`1,072 ms` p50 / `1,804 ms` p95** (`6.5×` slower p50 than `openai-decisions:gpt-6-luna`'s `164 ms`), `270,853` input + `2,610` output tokens, **`$0.02839`** total (`3.7×` costlier than `gpt-6-luna`'s `$0.00765`, and each distinct schema also incurs cold-grammar compilation overhead on dynamic schemas).
- **Multimodal & `ModelFallback` Role (`[Measured]`)**:
  - Red/Blue 16×16 PNG probe (`claude-haiku-5-5`): **`2/2`** (`1,934.9 ms` / `760.9 ms`).
  - **Ideal System-2 / `ModelFallback` Escalation Target (`[Measured]`)**: Paired behind `openai-decisions:gpt-6-luna` via `ModelFallback('anthropic:claude-haiku-5-5')`, routine tickets stayed on `gpt-6-luna` (`235 ms`), while complex disputes requiring a free-form `rationale: str` escalated via `UnfillableRoute` and returned validated `EscalatedTriage` from `claude-haiku-5-5` (`3,354 ms`).

## Benchmarks & Evidence

- **Local Live Harness (`benchmarks/cloud_decisions_2026-11-26_123548_openai_anthropic_logfire.md`)** `[Measured]`: Full 231-item JevBench, perturbations, smoke parity, multimodal PNG, and `ModelFallback` verified via `pydantic-ai-slim 2.75.0` with Logfire export (`https://logfire-us.pydantic.dev/mdfranz/dm-testing`).
- **Artificial Analysis Decision Index** `[Benchmark]`: Claude Haiku 5.5 variants score competitively (`~61–77` across index versions depending on configuration) when evaluated as generative baselines.

## Caveats & Open Questions

- No calibrated probabilities (`Brier`/`ECE`) and no native multi-question bundling in a single `/v1/decisions` pass (`bundle` requires packing fields into a single composite Pydantic schema).
- Dynamic per-request schemas pay grammar compilation + autoregressive output latency (`~1.07 s` median vs. `164 ms` on `openai-decisions:gpt-6-luna`), and default `pydantic-ai` calls do not insert `CachePoint` markers automatically.
- `claude-sonnet-5-5` costs **`20×`** more per token than `claude-haiku-5-5` (`$2/$10` vs. `$0.10/$0.50` per 1M); reserve Sonnet 5.5 for complex System-2 escalations (`ModelFallback`) rather than high-volume routing.

## Sources

- `[2026-11-26]` Installed source: `pydantic_ai/models/anthropic/`, `profiles/anthropic.py`, `genai-prices 0.2.5` — `[Primary]`
- `[2026-11-26]` Anthropic Claude 5.5 pricing, `output_config.format` structured outputs, & Adaptive Thinking docs — `[Primary]`
- `[2026-11-26]` Local benchmark (`benchmarks/cloud_decisions_2026-11-26_123548_openai_anthropic_logfire.md`) — `[Measured]`
