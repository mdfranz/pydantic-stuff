# pydantic-ai × Ollama Decision Models — Support Review & Upstream Candidates

**Date**: `2026-11-26` · **pydantic-ai-slim**: `2.75.0` · **Ollama**: `0.40.18` · **Host**: Mac Mini M4 24GB

## Current Support (what exists in `pydantic-ai-slim` 2.75.0)

| Piece | Status | Notes |
| --- | --- | --- |
| `DecisionModel` base (`models/decision.py`) | Built in | Maps `output_type` fields → typed questions: `bool` → `noul`, `Literal`/`Enum` → `choice`, 0..n ints with per-level descriptions → `score`. Tools/unions add a route question. `[Primary]` |
| `SystemOneModel` + `SystemOneProvider` (`system-one:`) | Built in | Speaks `POST /v1/systemone`; docstring names Ollama at `localhost:11434`. Needs `SYSTEM_ONE_BASE_URL` (no default). `[Primary]` |
| `OpenAIDecisionsModel`, `GoogleDecisionsModel`, `TypeSafeModel` | Built in | Other decision backends; not Ollama. `[Primary]` |
| `OllamaProvider` (`ollama:`) | Built in | Generative only (OpenAI-compatible `/v1/chat/completions`); does not route to decision models. `[Primary]` |

**Correction**: an earlier note in this session said `SystemOneModel` did not exist; that was wrong (a failed module scan was misread). The matrix's "Out-of-the-box (`SystemOneModel`)" verdict is **confirmed for text** `[Measured]`.

## Measured: built-in `SystemOneModel` vs. raw `/v1/systemone`

Same 6 cases / 8 questions as `benchmarks/head_to_head.py`. Built-in run: `benchmarks/pydantic_ai_official_systemone_2026-11-26_120519.json`. Raw-payload pydantic-ai wrapper run: `benchmarks/pydantic_ai_systemone_2026-11-26_115901.json`.

| Model | Raw payload (argmax) | Built-in `SystemOneModel` | Warm median (built-in) |
| --- | ---: | ---: | ---: |
| `laya:latest` | 8/8 | 8/8 | 49 ms |
| `d1:2b-mxfp8` | 8/8 | **7/8** (severity) | 136 ms |
| `strands-decider:2b-mxfp8` | 8/8 | **6/8** (urgency, severity) | 173 ms |
| `d3:4b-mxfp8` | 8/8 | 8/8 | 432 ms |
| `d3:9b-nvfp4` | 8/8 | **7/8** (severity) | 757 ms |
| `kev:9b-nvfp4` | 8/8 | 8/8 | 465 ms |
| `nimble-v3:9b-nvfp4` | 8/8 | 8/8 | 987 ms |

All misses are rubric (`score`) questions; two causes, both verified on the wire `[Measured]`:

1. **Rubric read rule**: pydantic-ai sets the field to `round(expected score)` (`models/decision.py:1448`), not the most-probable level. E.g. `d1` severity: probs `{0: .08, 1: .37, 2: .54}`, score `1.459` → **1**. The original harness used argmax → 2. Models with flat rubric distributions (`strands-decider`, `d3:9b`) are most affected.
2. **Different prompt**: pydantic-ai sends `instructions` as an object (`{"field": ..., "question": ...}`) rather than a plain string; Ollama accepts it but scores shift (`d1` severity score 1.55 → 1.46).

Neither is a bug. They are design choices. Benchmarks should state which read rule they use.

## Ollama `/v1/systemone` behavior probed locally `[Measured]`

- **Images supported**: top-level `"images": ["<base64>"]` (same as Ollama native API). `d1:2b-mxfp8` P(red) = `0.999` on a red PNG vs `0.003` on blue. Text-only `laya` → `{"error": "image inputs are not supported by this decision model"}`. Bad base64 → explicit unmarshal error.
- **`keep_alive` honored**: `"30s"` sets expiry; `0` unloads. Reachable today via `extra_body`.
- **Extra response field**: `prompt_eval_cached_count` (dropped by pydantic-ai's response parser).
- **Update (same day)**: Ollama does document the endpoint — [docs.ollama.com/api/systemone.md](https://docs.ollama.com/api/systemone.md) and [capabilities/decision.md](https://docs.ollama.com/capabilities/decision.md). `images` (base64 PNG/JPEG/WebP, vision models; WAV for audio models) and `keep_alive` are **official request fields** `[Primary]`; `prompt_eval_cached_count` is not in the documented schema. This makes candidate #1 a documented-API gap, not an Ollama quirk.

## Upstream Contribution Candidates (ranked)

| # | Change | Value | Effort / Risk | Precedent |
| --- | --- | --- | --- | --- |
| 1 | **Image input for `SystemOneModel`** — map `BinaryContent`/`ImageUrl` to base64 `images`; stop the blanket `UserError`; let the server reject text-only models (per repo rule: don't pre-emptively guard). | Unlocks multimodal `d1`, `d3` (2b/4b/9b) through pydantic-ai. Currently impossible without a custom model. | Medium. `images` is in Ollama's documented `/v1/systemone` schema; still confirm other System One servers accept it (may need a profile flag). Needs tests + `test_model_settings_support`. | Issue #11316 (image input for `drex-max`) |
| 2 | **Docs: Ollama as a System One server** — `SystemOneProvider(base_url="http://localhost:11434")`, `keep_alive` via `extra_body`, rubric read rule. | Low-friction; the docstring mentions Ollama but there's no setup guide. | Low. | Issues #10441, #14100 ("list X as a System One provider") |
| 3 | **Surface `prompt_eval_cached_count`** in `ModelResponse.provider_details`. | Cache-hit observability in Logfire for repeated states. | Low. Matches repo rule to expose provider data via `provider_details`. | — |
| 4 | **Ollama-aware base URL default** — fall back to `OLLAMA_BASE_URL` / `localhost:11434`, or an `ollama-system-one:` prefix. | Ergonomics. | Low–Medium; maintainers may prefer explicit config. | Issue #11316 (`nace:` prefix) |
| 5 | **Per-model decision limits** (`decision_max_choice_options`: `kev` 255, `tev1` 24, `autotrust-jev` 16, `laya` ~20) in `SystemOneProvider.model_profile()`. | Client-side refusal before an over-limit request. | Medium–High; Ollama tags are user-renameable, so name matching is fragile. Likely better left to `profile=`. | — |
| — | Rubric read rule (expected-score vs. mode) | — | Don't propose a change; behavior is documented ("output comes from `score` alone"). At most, a doc note. | — |

**Recommendation**: Lead with **#1** (clear functional gap, measured evidence, existing precedent), bundle **#2** docs, and offer **#3** as a small separate PR. Open an issue for #1 first to confirm the `images` wire shape with maintainers before coding.

## Sources

- `[2026-11-26]` Installed source: `pydantic_ai/models/system_one.py`, `providers/system_one.py`, `models/decision.py`, `profiles/decision.py` (`pydantic-ai-slim` 2.75.0) — `[Primary]`
- `[2026-11-26]` [pydantic-ai #10647](https://github.com/pydantic/pydantic-ai/issues/10647) — instructions object vs. string; notes `/v1/systemone` takes an object natively — `[Community]`
- `[2026-11-26]` [pydantic-ai #11316](https://github.com/pydantic/pydantic-ai/issues/11316), [#10441](https://github.com/pydantic/pydantic-ai/issues/10441), [#14100](https://github.com/pydantic/pydantic-ai/issues/14100), [#9765](https://github.com/pydantic/pydantic-ai/issues/9765), [#8552](https://github.com/pydantic/pydantic-ai/issues/8552) — open System One provider/image requests — `[Community]`
- `[2026-11-26]` [Ollama API docs](https://ollama.readthedocs.io/en/api/), [Ollama FAQ](https://docs.ollama.com/faq) — `images` / `keep_alive` on native API (no `/v1/systemone` doc found) — `[Primary]`
