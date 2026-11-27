# Cloud Decision Models Comparison (`pydantic-ai-slim 2.75.0`)

Master comparison of **Gemini**, **OpenAI**, and **Anthropic** models for structured decision workloads (`classification`, `routing`, `rubrics`, `guardrails`, `policy checks`) via `pydantic-ai`, alongside measured local **Mac Mini M4 (24GB)** Ollama `/v1/systemone` baselines.

## 1. Cloud Provider & Model Matrix (`[Measured]` & `[Primary]`)

| Model (`pydantic-ai` ID) | API / Mode | Pricing (In / Cache / Out per 1M) `[Pri]` | Ctx | Native Probs & Calib. | Caps (Q / Choice / Score) `[Pri]` | `pydantic-ai` Status | Smoke (`8Q`) `[Mea]` | JevBench Acc / CC (`231`) `[Mea]` | Brier ↓ / ECE ↓ `[Mea]` | p50 / p95 ms `[Mea]` | Cost (`231` items) `[Mea]` | Best For |
| --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `google-decisions:gemini-decision-1` | `POST v1beta/decisions` (`GoogleDecisionsModel`) | `$0.10 / — / $0.00` | 256K | **Yes** (`noul`, `choice`, `score`) | **`1000` / `255` / `255`** | `Minor config` (client blocks PDF/audio/video) | **8/8** (`312 ms`) | **`92.2%` / `88.6%`** | **`0.118` / `0.035`** | `230 / 462` | `$0.01585` (`0` cache hits) | **Best Overall Accuracy, Calibration (Brier) & Perturbation Invariance** |
| `openai-decisions:gpt-6-luna` | `POST /v1/decisions` (`OpenAIDecisionsModel`) | `$0.10 / $0.01 / $0.00` | ~1.05M | **Yes** (`predicate`, `choice`, `score`) | `200` (doc `400`) / `255` / `10` | `Out-of-the-box` | **8/8** (`252 ms`) | `87.9%` / `82.2%` | `0.164` / **`0.026`** | **`164 / 512`** | **`$0.00765`** (`56%` auto-cached) | **Lowest Cost/Latency Cloud Router & `N=255` Cardinality** |
| `anthropic:claude-haiku-5-5` | `POST /v1/messages` (`NativeOutput` + `thinking=False`) | `$0.10 / $0.01* / $0.50` *(≤100K)* | 200K+ | **No** (discrete JSON only) | Schema-bounded | `Out-of-the-box` (omit `temp`) | **8/8** (`1014 ms`) | `89.2%` / `84.1%` | `— / —` | `1072 / 1804` | `$0.02839` (`0` cache hits*) | **Top `temporal_numeric`, Free-Form Reasoning & `ModelFallback` Target** |
| `openai:gpt-6-luna` | `POST /v1/responses` (`NativeOutput` control) | `$0.10 / $0.01 / $0.50` | ~1.05M | **No** (discrete JSON only) | Schema-bounded | `Out-of-the-box` | **8/8** (`868 ms`) | — | `— / —` | — | — | Generative baseline (`3.4×` slower than `/v1/decisions` on same weights) |
| `anthropic:claude-sonnet-5-5` | `POST /v1/messages` (`NativeOutput` + `thinking=False`) | `$2.00 / $0.10* / $10.00` | 200K+ | **No** (discrete JSON only) | Schema-bounded | `Out-of-the-box` (omit `temp`) | **8/8** (`911 ms`) | — | `— / —` | — | — | Heavy System-2 `ModelFallback` escalation (`20×` Haiku cost) |

*\*Anthropic prompt caching requires explicit `CachePoint` markers (`0` automatic cache reads observed in default JevBench run); OpenAI `/v1/decisions` cached `56.0%` (`86,437 / 154,317` tokens) automatically.*

---

## 2. Robustness & Option-Count Scaling (`[Measured]`)

| Model | Order Flip (Rev / Rot) ↓ | Anon-Label Flip ↓ | None-Option Acc ↑ | Polarity Swap Followed ↑ | Bundle Flip ↓ (`Δp`) | Repeat Flip ↓ (`Δp`) | `N=8` | `N=32` | `N=64` | `N=128` | `N=255` (`p50 ms`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `google-decisions:gemini-decision-1` | **0.0% / 0.0%** | **0.0%** | **86.7%** | **100.0%** | **0.0%** (`0.000`) | **0.0%** (`0.000`) | **100.0%** | **100.0%** | **100.0%** | 91.7% | 83.3% (`284 ms`) |
| `openai-decisions:gpt-6-luna` | **0.0% / 0.0%** | **0.0%** | 66.7% | 63.9% | **0.0%** (`0.000`) | **0.0%** (`0.000`) | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **91.7%** (`192 ms`) |
| `anthropic:claude-haiku-5-5` | **0.0% / 0.0%** | 1.4% | 85.0% | 88.9% | N/A | **0.0%** (`—`) | — | — | — | — | — |

---

## 3. Cloud vs. Local Mac Mini M4 (24GB Ollama `/v1/systemone`) `[Measured]`

Pulling directly from `benchmarks/decision_bench_v2_2026-11-26_121500.md`, `benchmarks/pydantic_ai_official_systemone_2026-11-26_120519.json`, `benchmarks/cloud_decisions_2026-11-26_123548_openai_anthropic_logfire.md`, and `benchmarks/cloud_decisions_2026-11-26_132008_gemini_logfire.md`:

| Tier / Model | Where It Runs | Official `pydantic-ai` Smoke (`8Q`) | JevBench Acc (`231`) | Hard Acc (`111`) | Brier ↓ / ECE ↓ | `none_option` / `polarity_swap` | `N=255` Acc | p50 / p95 ms | Var. Cost (`231` items) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `google-decisions:gemini-decision-1` | Cloud (`v1beta/decisions`) | **8/8** | **92.2%** | **83.8%** | **`0.118` / `0.035`** | **86.7% / 100.0%** | 83.3% | `230 / 462` | `$0.01585` |
| `anthropic:claude-haiku-5-5` | Cloud (`Messages`) | **8/8** | 89.2% | 79.3% | `— / —` | 85.0% / 88.9% | — | `1072 / 1804` | `$0.02839` |
| `openai-decisions:gpt-6-luna` | Cloud (`/v1/decisions`) | **8/8** | 87.9% | 78.4% | `0.164` / **`0.026`** | 66.7% / 63.9% | **91.7%** | `164 / 512` | `$0.00765` |
| `d3:4b-mxfp8` | Local M4 (`5.51 GB`) | **8/8** | 81.4% | 61.3% | `0.242 / 0.076` | 81.7% / **100.0%** | 75.0% | `548 / 7598` | `$0` (hardware) |
| `strands-decider:2b-mxfp8` | Local M4 (`2.0 GB`) | 6/8 *(pai-round)* | 79.2% | 62.2% | `0.280 / 0.058` | 83.3% / 97.2% | 0.0% | `197 / 2904` | `$0` (hardware) |
| `kev:9b-nvfp4` | Local M4 (`6.45 GB`) | **8/8** | 78.8% | 59.5% | `0.301 / 0.061` | 86.7% / 66.7% | 75.0% | `605 / 12607` | `$0` (hardware) |
| `d1:2b-mxfp8` | Local M4 (`2.45 GB`) | 7/8 *(pai-round)* | 76.2% | 54.1% | `0.284 / 0.075` | 83.3% / **100.0%** | 0.0% | `166 / 2849` | `$0` (hardware) |
| `laya:latest` | Local M4 (`0.9 GB`) | **8/8** | 58.9% | 35.1% | `0.530 / 0.085` | **96.7%** / 16.7% | 0.0% | **`53 / 162`** | `$0` (hardware) |

**Takeaway**: On `Original` (`91.7–100%`) and `Easy` (`100%`) items, local M4 System One models (`d1:2b`, `strands-decider:2b`, `d3:4b`) match or beat cloud APIs on routine privacy/zero-variable-cost routing (`53–548 ms` p50). Cloud Decision APIs pull decisively ahead on **Hard multi-hop/long-policy accuracy** (`83.8%` on `gemini-decision-1` vs. `62.2%` best measured local), **calibration** (`Brier 0.118–0.164` vs. `0.242+`), **tail latency** (`462–512 ms` p95 vs. `2.8–12.6 s`), and **`N=255` scaling** (`91.7%` @ `192 ms` on `gpt-6-luna` vs. `75.0%` @ `13–21 s` locally).

---

## 4. `pydantic-ai 2.75.0` Compatibility & Findings

| Provider / Model | Key Observations & Gaps (`[Primary]` / `[Measured]`) | Workaround / Best Practice |
| --- | --- | --- |
| **Google (`GoogleDecisionsModel`)** | 1) `decide()` blocks non-image `BinaryContent` (`PDF`/`audio`/`video`) with `UserError`, even though the Gemini Decisions API accepts them natively.<br>2) `google-genai<3` lacks typed `decisions` helper (uses private `_api_client.async_request`).<br>3) `genai-prices 0.2.5` lacks `google-decisions` pricing.<br>4) Account-bound Vertex Express keys fail with `403 API_KEY_SERVICE_BLOCKED` unless `Gemini API` is enabled (standard AI Studio keys work cleanly). | Use a standard Gemini Developer API key; extract text/page images for PDFs until upstream lifts the `image.is_image` check. |
| **OpenAI (`OpenAIDecisionsModel`)** | 1) `max_questions` is capped at `200` client-side (OpenAI docs list `400`).<br>2) Ambiguous nonsense input returned `"type": "refusal"` -> `ContentFilterError` on `UnsureRoute` probe.<br>3) Union `output_type=[A, B]` requires class docstrings on every model (`UserError` otherwise). | Use default `ModelFallback(..., fallback_on=(ModelAPIError, ContentFilterError, OutputTokenLimitExceeded))` so both `DecisionHandOff` and API refusals escalate cleanly; always document union route classes. |
| **Anthropic (`AnthropicModel`)** | 1) No `/v1/decisions` endpoint; `claude-*-5-5` drops `temperature` with `UserWarning`.<br>2) `0` automatic cache reads without explicit `CachePoint`. | Use `NativeOutput(Schema)` + `ModelSettings(thinking=False)` (no `temperature`), and add `CachePoint` on repeated long-policy prefixes. |

---

## 5. Shortlists by Use Case

- **Best Overall Cloud Decision Model (Accuracy, Calibration & Invariance)**: **`google-decisions:gemini-decision-1`** (`92.2%` JevBench, `100%` `Original`/`Easy`/`para`/`polarity_swap`, `0.118` Brier, `0.0%` order/anon/bundle/repeat flips, clean `UnfillableRoute` & `UnsureRoute` handoffs).
- **Best Low-Cost / Low-Latency Cloud Router & `N=255` Cardinality**: **`openai-decisions:gpt-6-luna`** (`164 ms` p50, `0.026` ECE, **`91.7%` at `N=255`**, **`$0.00765`** for 231 items via automatic prompt caching; verify negative gates since `none_option` is `66.7%` and `polarity_swap` is `63.9%`).
- **Best Generative Escalation Target (`ModelFallback`) & `temporal_numeric`**: **`anthropic:claude-haiku-5-5`** (`89.2%` JevBench, `46.7%` `temporal_numeric`, `85.0%` `none_option`, `88.9%` `polarity_swap`) paired behind `gemini-decision-1` or `gpt-6-luna` via `ModelFallback` so routine routing costs `$0` output and `~225–235 ms`, while free-form rationales and low-confidence routes escalate to Claude.
