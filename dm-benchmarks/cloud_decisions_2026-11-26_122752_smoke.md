# Cloud Decision-Model Benchmark (`2026-11-26 12:27:52 -0500`)

Harness: `pai-poc/cloud_decisions_bench.py` (`pydantic-ai-slim 2.75.0` + Logfire). Data: JevBench public set (`benchmarks/data/jevbench`, MIT). `[Measured]`

## Smoke Parity Suite (6 cases / 8 questions, `official_systemone_bench.py` equivalent)

| Model | Score | Warm Median ms | Input Tokens | Output Tokens | Est. Cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `google-decisions:gemini-decision-1` | 0/8 | 140 | 0 | 0 | $0.000000 |
| `openai-decisions:gpt-6-luna` | 8/8 | 246 | 1461 | 0 | $0.000146 |
| `openai:gpt-6-luna` | 8/8 | 961 | 723 | 96 | $0.000120 |
| `anthropic:claude-haiku-5-5` | 8/8 | 2003 | 2098 | 74 | $0.000247 |
| `anthropic:claude-sonnet-5-5` | 8/8 | 1072 | 2098 | 72 | $0.004916 |

## Multimodal & Client Gap Probes

| Model | Red PNG | Blue PNG | Extra Check |
| --- | --- | --- | --- |
| `google-decisions:gemini-decision-1` | ERR: ModelAuthenticationError: status_code: 403, model_name: gemini-decision-1, body: {'error': {'code': 403, 'message': 'Requests to this API generativelanguage.googleapis.com method genai.gemini.v1beta.DecisionService.CreateDecis | ERR: ModelAuthenticationError: status_code: 403, model_name: gemini-decision-1, body: {'error': {'code': 403, 'message': 'Requests to this API generativelanguage.googleapis.com method genai.gemini.v1beta.DecisionService.CreateDecis | UserError: A user prompt contains an unsupported file: this model accepts text and images only. |
| `openai-decisions:gpt-6-luna` | ✓ (True, 233.6 ms) | ✓ (False, 208.6 ms) | — |
| `anthropic:claude-haiku-5-5` | ✓ (True, 2505.5 ms) | ✓ (False, 850.6 ms) | — |

## `ModelFallback` (`DecisionHandOff` -> Claude Haiku 5.5)

- `google-decisions:gemini-decision-1` → `anthropic:claude-haiku-5-5`:
  - `fast_stays_on_decision`: `{"error": "UserError: A decision model weighs each route by what it is for, and 'FastTriage' says nothing about itself. Give the output type a docstring that says what filling it does."}`
  - `complex_escalates_unfillable`: `{"error": "UserError: A decision model weighs each route by what it is for, and 'FastTriage' says nothing about itself. Give the output type a docstring that says what filling it does."}`
  - `unsure_threshold_escalates`: `{"error": "UserError: A decision model weighs each route by what it is for, and 'FastTriage' says nothing about itself. Give the output type a docstring that says what filling it does."}`
- `openai-decisions:gpt-6-luna` → `anthropic:claude-haiku-5-5`:
  - `fast_stays_on_decision`: `{"error": "UserError: A decision model weighs each route by what it is for, and 'FastTriage' says nothing about itself. Give the output type a docstring that says what filling it does."}`
  - `complex_escalates_unfillable`: `{"error": "UserError: A decision model weighs each route by what it is for, and 'FastTriage' says nothing about itself. Give the output type a docstring that says what filling it does."}`
  - `unsure_threshold_escalates`: `{"error": "UserError: A decision model weighs each route by what it is for, and 'FastTriage' says nothing about itself. Give the output type a docstring that says what filling it does."}`
