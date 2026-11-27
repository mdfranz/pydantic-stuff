# Cloud Decision-Model Benchmark (`2026-11-26 13:20:08 -0500`)

Harness: `pai-poc/cloud_decisions_bench.py` (`pydantic-ai-slim 2.75.0` + Logfire). Data: JevBench public set (`benchmarks/data/jevbench`, MIT). `[Measured]`

## Smoke Parity Suite (6 cases / 8 questions, `official_systemone_bench.py` equivalent)

| Model | Score | Warm Median ms | Input Tokens | Output Tokens | Est. Cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `google-decisions:gemini-decision-1` | 8/8 | 312 | 1048 | 0 | $0.000105 |

## JevBench Public Suite (231 items)

| Model | Coverage | Acc | Acc (valid) | Chance-corr. | Original | Easy | Hard | Brier ↓ | ECE ↓ | Ord. MAE ↓ | Score argmax / pai-round | Para same / both ✓ | Trap ↓ | p50 / p95 ms | Tokens (In / Out) | Est. Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `google-decisions:gemini-decision-1` | 100.0 | 92.2 | 92.2 | 88.6 | 100.0 | 100.0 | 83.8 | 0.118 | 0.035 | 0.26 | 83.3 / 88.9 | 100.0 / 100.0 | 10.4 | 230 / 462 | 158532 / 0 | $0.01585 |

### Hard tier by family (accuracy %)

| Model | adversarial | ambiguous | judge_hard | long_policy | multi_hop | probability | routing_hard | temporal_numeric | tradeoff | trap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `google-decisions:gemini-decision-1` | 100.0 | 100.0 | 94.1 | 84.2 | 77.8 | 100.0 | 100.0 | 33.3 | 100.0 | 100.0 |

## Robustness (perturbations of original + easy items)

| Model | Order flip (rev / rot) ↓ | Anon-label flip ↓ | None-option acc ↑ | Polarity swap followed ↑ | Bundle flip ↓ / Δp | Repeat flip ↓ / Δp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `google-decisions:gemini-decision-1` | 0.0 / 0.0 | 0.0 | 86.7 | 100.0 | 0.0 / 0.000 | 0.0 / 0.000 |

## Option-count scaling (12 intent/routing items)

| Model | N=8 | N=32 | N=64 | N=128 | N=255 | p50 ms @ N=255 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `google-decisions:gemini-decision-1` | 100.0 (12/12) | 100.0 (12/12) | 100.0 (12/12) | 91.7 (12/12) | 83.3 (12/12) | 284 |

## Multimodal & Client Gap Probes

| Model | Red PNG | Blue PNG | Extra Check |
| --- | --- | --- | --- |
| `google-decisions:gemini-decision-1` | ✓ (True, 865.7 ms) | ✓ (False, 504.2 ms) | UserError: A user prompt contains an unsupported file: this model accepts text and images only. |

## `ModelFallback` (`DecisionHandOff` -> Claude Haiku 5.5)

- `google-decisions:gemini-decision-1` → `anthropic:claude-haiku-5-5`:
  - `fast_stays_on_decision`: `{"output_type": "FastTriage", "output": {"team": "shipping"}, "models_used": ["gemini-decision-1"], "ms": 225.4, "error": null}`
  - `complex_escalates_unfillable`: `{"output_type": "EscalatedTriage", "output": {"team": "legal", "rationale": "The customer reports that a defective charger burned an office outlet and that their counsel is preparing a formal liability claim. This involves a safety incident and a legal dispute, which requires escalation to the legal team with a written rationale rather than routine single-issue handling."}, "models_used": ["claude-haiku-5-5"], "ms": 2866.5, "error": null}`
  - `unsure_threshold_escalates`: `{"output_type": "FastTriage", "models_used": ["claude-haiku-5-5"], "ms": 2381.6, "error": null}`
