# Cloud Decision-Model Benchmark (`2026-11-26 12:35:48 -0500`)

Harness: `pai-poc/cloud_decisions_bench.py` (`pydantic-ai-slim 2.75.0` + Logfire). Data: JevBench public set (`benchmarks/data/jevbench`, MIT). `[Measured]`

## Smoke Parity Suite (6 cases / 8 questions, `official_systemone_bench.py` equivalent)

| Model | Score | Warm Median ms | Input Tokens | Output Tokens | Est. Cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `openai-decisions:gpt-6-luna` | 8/8 | 252 | 1461 | 0 | $0.000146 |
| `openai:gpt-6-luna` | 8/8 | 868 | 723 | 96 | $0.000120 |
| `anthropic:claude-haiku-5-5` | 8/8 | 1014 | 2098 | 75 | $0.000247 |
| `anthropic:claude-sonnet-5-5` | 8/8 | 911 | 2098 | 72 | $0.004916 |

## JevBench Public Suite (231 items)

| Model | Coverage | Acc | Acc (valid) | Chance-corr. | Original | Easy | Hard | Brier ↓ | ECE ↓ | Ord. MAE ↓ | Score argmax / pai-round | Para same / both ✓ | Trap ↓ | p50 / p95 ms | Tokens (In / Out) | Est. Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `openai-decisions:gpt-6-luna` | 100.0 | 87.9 | 87.9 | 82.2 | 94.4 | 100.0 | 78.4 | 0.164 | 0.026 | 0.18 | 88.9 / 88.9 | 88.9 / 88.9 | 14.2 | 164 / 512 | 154317 / 0 | $0.00765 |
| `anthropic:claude-haiku-5-5` | 100.0 | 89.2 | 89.2 | 84.1 | 97.2 | 100.0 | 79.3 | — | — | 0.11 | 88.9 / 88.9 | 94.4 / 94.4 | 14.2 | 1072 / 1804 | 270853 / 2610 | $0.02839 |

### Hard tier by family (accuracy %)

| Model | adversarial | ambiguous | judge_hard | long_policy | multi_hop | probability | routing_hard | temporal_numeric | tradeoff | trap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `openai-decisions:gpt-6-luna` | 100.0 | 100.0 | 94.1 | 73.7 | 83.3 | 70.0 | 100.0 | 26.7 | 83.3 | 100.0 |
| `anthropic:claude-haiku-5-5` | 100.0 | 71.4 | 88.2 | 78.9 | 88.9 | 70.0 | 100.0 | 46.7 | 66.7 | 100.0 |

## Robustness (perturbations of original + easy items)

| Model | Order flip (rev / rot) ↓ | Anon-label flip ↓ | None-option acc ↑ | Polarity swap followed ↑ | Bundle flip ↓ / Δp | Repeat flip ↓ / Δp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `openai-decisions:gpt-6-luna` | 0.0 / 0.0 | 0.0 | 66.7 | 63.9 | 0.0 / 0.000 | 0.0 / 0.000 |
| `anthropic:claude-haiku-5-5` | 0.0 / 0.0 | 1.4 | 85.0 | 88.9 | — / — | 0.0 / — |

## Option-count scaling (12 intent/routing items)

| Model | N=8 | N=32 | N=64 | N=128 | N=255 | p50 ms @ N=255 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `openai-decisions:gpt-6-luna` | 100.0 (12/12) | 100.0 (12/12) | 100.0 (12/12) | 100.0 (12/12) | 91.7 (12/12) | 192 |

## Multimodal & Client Gap Probes

| Model | Red PNG | Blue PNG | Extra Check |
| --- | --- | --- | --- |
| `openai-decisions:gpt-6-luna` | ✓ (True, 330.3 ms) | ✓ (False, 130.8 ms) | — |
| `anthropic:claude-haiku-5-5` | ✓ (True, 1934.9 ms) | ✓ (False, 760.9 ms) | — |

## `ModelFallback` (`DecisionHandOff` -> Claude Haiku 5.5)

- `openai-decisions:gpt-6-luna` → `anthropic:claude-haiku-5-5`:
  - `fast_stays_on_decision`: `{"output_type": "FastTriage", "output": {"team": "shipping"}, "models_used": ["gpt-6-luna"], "ms": 235.0, "error": null}`
  - `complex_escalates_unfillable`: `{"output_type": "EscalatedTriage", "output": {"team": "legal", "rationale": "Customer reports a defective charger that burned an office outlet and states their counsel is preparing a formal liability claim. This involves a potential safety incident and a legal dispute, which requires legal review. The matter is also likely to involve multiple parties (billing, technical root-cause analysis of the charger, and possibly shipping/return records), so it is not a routine single-issue ticket and requires escalation with a written rationale."}, "models_used": ["claude-haiku-5-5"], "ms": 3354.1, "error": null}`
  - `unsure_threshold_escalates`: `{"error": "ContentFilterError: Content filter triggered. The OpenAI Decisions API declined to answer: 'FastTriage.team', body:\n{\n  \"model\": \"gpt-6-luna\",\n  \"answers\": [\n    {\n      \"type\": \"refusal\",\n      \"name\": \"FastTriage.team\"\n    },\n    {\n      \"type\": \"choice\",\n  "}`
