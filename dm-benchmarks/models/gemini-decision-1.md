# Google Gemini Decision 1 (`gemini-decision-1`)

## Metadata

| Field | Value |
| --- | --- |
| **Provider / Model ID** | `google-decisions:gemini-decision-1` (`gemini-decision-1`) |
| **API Endpoint** | `POST https://generativelanguage.googleapis.com/v1beta/decisions` (`v1beta`) `[Primary]` |
| **Architecture** | Dedicated closed-set multimodal Decision Model (zero output text tokens; parallel multi-question prefill pass) `[Primary]` |
| **Context Window** | `262,144` tokens (256K shared across input + all questions) `[Primary]` |
| **Pricing (USD)** | `$0.10 / 1M` input tokens · **`$0.00` output tokens** · free prototyping tier in AI Studio `[Primary]` |
| **pydantic-ai Readiness** | `Minor config` (`GoogleDecisionsModel`, `google-decisions:gemini-decision-1`, `pydantic-ai-slim[google]`; client blocks PDF/audio/video) `[Measured]` |
| **Status** | `Shortlisted (Top Accuracy & Perturbation Robustness Cloud Decision Model)` |
| **Last Updated** | `2026-11-26` |

## API Limits, Pricing & Throughput Profile

- **Modalities Supported (API)**: Text, images, audio, video, and PDFs natively in a single forward pass `[Primary]`.
- **Hard Request Limits (verified in `pydantic_ai/models/google_decisions.py`)** `[Primary]`:
  - `max_questions = 1000` per request (1001st returns HTTP 400).
  - `max_choice_options = 255` per `choice` question (256th returns HTTP 400).
  - `max_score_levels = 255` per `score` rubric (256th returns HTTP 400; 2–10 recommended `[Primary]`).
  - `requires_instructions = True` (API rejects empty `instructions` on any question).
- **Latency (`[Measured]` vs. `[Primary]`)**: Published `150–250 ms` for `≤16K` tokens `[Primary]`; measured **`230 ms` p50 / `462 ms` p95** across 231 JevBench items (`concurrency=5`) and **`312 ms`** warm median on the 6-case `smoke` suite `[Measured]`.
- **Shared-Input Economics (`[Measured]`)**: `158,532` input tokens / `0` output tokens (`0` cache-read tokens reported) across 231 JevBench items = **`$0.01585`** (`~$0.000069` per decision). Evaluating up to `1,000` questions in one request bills the shared context once `[Primary]`.

## Decision-Model Assessment

- **Decision Quality & Determinism (`[Measured]`)**:
  - **JevBench Public (231 items)**: **`92.2%`** accuracy (`213/231`), **`88.6%`** chance-corrected accuracy, `100.0%` coverage (`0` errors) — **highest overall accuracy of all tested cloud and local models**, matching Google's published `92.2%` launch figure `[Primary]`/`[Measured]`.
  - **Tier Breakdown**: `Original` **`100.0%` (`72/72`)**, `Easy` **`100.0%` (`48/48`)**, `Hard` **`83.8%` (`93/111`)**.
  - **Calibration & Consistency**: **Brier `0.118`** (best measured), **ECE `0.035`**, **Ordinal MAE `0.26`** (`score` argmax `83.3%` / `pai-round` `88.9%`), **Paraphrase consistency `100.0%` (`100.0%` both ✓)**, **Trap rate `10.4%`** (lowest measured).
  - **Hard-Tier Families**: `100.0%` on `adversarial`, `ambiguous`, `probability`, `routing_hard`, `tradeoff`, and `trap`; `94.1%` `judge_hard`; `84.2%` `long_policy`; `77.8%` `multi_hop`; weak spot is `temporal_numeric` (`33.3%`, though still above `gpt-6-luna`'s `26.7%`).
- **Perturbation Robustness (`[Measured]`)**:
  - **Perfect invariance** across order/label/bundle/repeat/polarity perturbations: **`0.0%` `order_reverse` flip (`0.000` Δp)**, **`0.0%` `order_rotate` flip (`0.000` Δp)**, **`0.0%` `anon_labels` flip**, **`0.0%` `bundle` flip (`0.000` Δp)**, **`0.0%` `repeat` flip (`0.000` Δp)**, **`100.0%` (`36/36`) `polarity_swap` consistency**, and **`86.7%` (`52/60`) `none_option` accuracy** (highest of all 3 cloud models).
- **Option-Count Scaling (`[Measured]`)**:
  - `N=8`: **`100.0%` (`12/12`, `176 ms`)** · `N=32`: **`100.0%` (`12/12`, `198 ms`)** · `N=64`: **`100.0%` (`12/12`, `207 ms`)** · `N=128`: `91.7%` (`11/12`, `256 ms`) · `N=255`: `83.3%` (`10/12`, `284 ms` p50). (Slightly trails `openai-decisions:gpt-6-luna`'s `100.0%` @ `N=128` / `91.7%` @ `N=255`.)
- **Tool Calling & `ModelFallback` (`[Measured]`)**:
  - Passed all 3 `ModelFallback` (`DecisionHandOff` → `anthropic:claude-haiku-5-5`) probes cleanly:
    - Routine ticket stayed on `gemini-decision-1` (`FastTriage`, **`225.4 ms`**, zero Anthropic call).
    - Legal dispute requiring free-form `rationale: str` triggered `UnfillableRoute` and handed off to `claude-haiku-5-5` (`EscalatedTriage`, `2,866.5 ms`).
    - Ambiguous input below `decision_route_threshold=0.999` triggered `UnsureRoute` and handed off cleanly to `claude-haiku-5-5` (`FastTriage`, `2,381.6 ms`) — without the API content-filter refusal observed on `gpt-6-luna`.
- **Operational & `pydantic-ai` Readiness**:
  - **Key gotcha**: Reads `GOOGLE_API_KEY` or `GEMINI_API_KEY` (`GOOGLE_API_KEY` takes precedence). Standard Gemini Developer API keys work; Vertex Express account-bound keys without `Gemini API` enabled return `403 API_KEY_SERVICE_BLOCKED` `[Measured]`.
  - **Client-side media gap (`[Measured]` & `[Primary]`)**: While Google's Decisions API accepts audio, video, and PDFs natively, `GoogleDecisionsModel.decide()` (`pydantic-ai-slim 2.75.0`, lines 182–183) enforces image-only `BinaryContent`: passing an `application/pdf` raised `UserError: A user prompt contains an unsupported file: this model accepts text and images only.` `[Measured]` Workaround: extract text/page images or call `client._api_client.async_request('post', 'decisions', ...)` directly until upstream lifts the check.
  - **Red/Blue 16×16 PNG probe (`[Measured]`)**: **`2/2`** (`True` in `865.7 ms`, `False` in `504.2 ms`).
  - **`genai-prices` & SDK gaps (`[Primary]`)**: `google-genai<3` lacks a typed `decisions` helper (uses `_api_client.async_request`), and `genai-prices 0.2.5` does not yet list `google-decisions` pricing.
  - **Union-route docstring rule (`[Measured]`)**: Union `output_type=[RouteA, RouteB]` requires docstrings on every route class or `DecisionModel` raises `UserError`.

## Benchmarks & Evidence

- **Local Live Harness (`benchmarks/cloud_decisions_2026-11-26_132008_gemini_logfire.md`)** `[Measured]`: Full `smoke` (`8/8`), `jevbench` (`231` items), `perturb`, `card` (`N=8..255`), `multimodal`, and `fallback` verified end-to-end via `pydantic-ai-slim 2.75.0` with Logfire export (`https://logfire-us.pydantic.dev/mdfranz/dm-testing`).
- **Google Launch Benchmarks (`2026-11-11`)** `[Primary]`: JevBench public `92.2%`, ECE `~0.025`, When2Call `79.9%`.
- **Artificial Analysis Decision Index** `[Benchmark]`: Score `~66` (ahead of `gpt-6-luna` at `~60`).

## Caveats & Open Questions

- `pydantic-ai-slim 2.75.0` blocks non-image multimodal inputs (PDF/audio/video) client-side on `GoogleDecisionsModel` despite server-side support, and `genai-prices 0.2.5` omits `google-decisions`.
- `temporal_numeric` hard-tier items remain weak (`33.3%`), so arithmetic/date-diff policy rules still benefit from `ModelFallback` or tool offloading.
- Slightly slower median latency (`230 ms` vs. `164 ms`) and higher uncached cost (`$0.01585` vs. `$0.00765` with OpenAI auto-caching) than `openai-decisions:gpt-6-luna`, and slightly lower accuracy at `N=255` (`83.3%` vs. `91.7%`).

## Sources

- `[2026-11-26]` Installed source: `pydantic_ai/models/google_decisions.py`, `providers/google.py` (`pydantic-ai-slim 2.75.0`) — `[Primary]`
- `[2026-11-26]` Google AI Studio Decisions API docs (`https://aistudio.google.com/docs/decisions-api`) & Nov 11, 2026 Google launch coverage — `[Primary]`
- `[2026-11-26]` Artificial Analysis Decision Index — `[Benchmark]`
- `[2026-11-26]` Local benchmark (`benchmarks/cloud_decisions_2026-11-26_132008_gemini_logfire.md`) — `[Measured]`
