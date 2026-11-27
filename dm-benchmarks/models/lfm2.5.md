# Liquid AI LFM2.5 (`lfm2.5`)

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `lfm2.5:8b` / `8b-a1b-q4_K_M` / `latest` (`9cf756159fc2`), `lfm2.5:8b-a1b-q8_0` (`4fa3787050ca`) *(plus `8b-a1b-bf16` 17 GB & `24b*` 14–48 GB — >10B / Too Large or Tight)* |
| **Parameters (Total / Active)** | `8.3B` / `1.5B` active (`LFM2.5-8B-A1B` MoE) |
| **Architecture** | Hybrid convolution + attention MoE (Text-only) |
| **Context Window (Max / Practical M4)** | `125K` (`128K` in `24b`) / `8k–32k` practical M4 |
| **Recommended Quantization** | `lfm2.5:8b-a1b-q4_K_M` (`5.2 GB`, GGUF) or `8b-a1b-q8_0` (`9.0 GB`, GGUF) |
| **License** | **LFM Open License v1.0** (free for research/non-commercial & commercial entities **< $10M annual revenue**; ≥ $10M commercial use is NOT licensed) |
| **pydantic-ai Readiness** | `Out-of-the-box` (`Minor config`: verify Ollama translates LFM2.5 Pythonic `tool_call_start` / `func(...)` syntax cleanly) |
| **Status** | `Candidate` (fast 1.5B-active MoE, but largely superseded by newer `lfm3:1.7b`, `qwen3.5`, and `granite4.2`) |
| **Last Updated** | `2026-11-26` |

## Ollama Tags & Mac Mini M4 (24GB) Fit

| Ollama Tag | Hash | Disk Size | Active Params | M4 24GB Fit | Notes |
| --- | --- | ---: | ---: | --- | --- |
| `lfm2.5:8b` / `8b-a1b-q4_K_M` / `latest` | `9cf756159fc2` | 5.2 GB | 1.5B | `Comfortable` | ~6.3–7.0 GB est. RAM |
| `lfm2.5:8b-a1b-q8_0` | `4fa3787050ca` | 9.0 GB | 1.5B | `Comfortable` | ~10.2–11.0 GB est. RAM |
| `lfm2.5:8b-a1b-bf16` | `c6d60dd0c9c3` | 17 GB | 1.5B | `Too Large` | Exceeds ~16–18 GB usable M4 ceiling |
| `lfm2.5:24b` / `24b-a2b-q4_K_M` | `00a755688d0f` | 15 GB | 2B | `Tight` (>10B) | 24B total (>10B cap); `24b-q8_0` (25 GB) & `bf16` (48 GB) `Too Large` |

- **Weight Size (Disk / Loaded)** `[Primary]`: `5.2 GB` (`8b-a1b-q4_K_M`), `9.0 GB` (`8b-a1b-q8_0`).
- **Est. Total RAM @ 8k / 16k / 32k** `[Estimated]`: **~6.3–7.2 GB** (`q4_K_M`) / **~10.2–11.2 GB** (`q8_0`) (`Comfortable`).
- **Fit Tier**: `Comfortable` (`8b-a1b-q4_K_M`, `8b-a1b-q8_0`).
- **Est. M4 Decode Speed** `[Estimated]` (MoE active-parameter bandwidth math: ~1.0–1.7 GB active weights/tok @ `Q4_K_M`/`Q8_0` at 120 GB/s): **~55–85 tok/s** (`8b-a1b-q4_K_M`), **~45–60 tok/s** (`8b-a1b-q8_0`).

## Decision-Model Assessment

- **Decision Quality & Determinism** `[Primary]`: Solid improvement over `LFM2-8B-A1B` (`IFEval 91.84`, `IFBench 56.47`, `AA-Omniscience non-hallucination 63.47%`), but **outperformed by the newer `LFM3-1.7B`** across `BFCL v4` (`71.39` vs `49.73`), `IFBench` (`73.03` vs `56.47`), `IFEval` (`96.66` vs `91.84`), and non-hallucination (`92.93%` vs `63.47%`) `[Primary]`.
- **Structured Output & JSON Schema** `[Primary]`: GGUF tags on Ollama support `llama.cpp` grammar-constrained `json_schema` for `NativeOutput`.
- **Tool & Function Calling** `[Primary]`: `BFCL v3 64.79`, `BFCL v4 49.73`, `Tau² Telecom 88.07 / Retail 39.82`. Uses Pythonic call syntax (`<|tool_call_start|>[func(arg="val")]<|tool_call_end|>`); verify Ollama `/v1` translates this cleanly into OpenAI `tool_calls` for `pydantic-ai` `ToolOutput`.
- **Thinking / Latency Tradeoff** `[Primary]`: Reasoning-tuned (`thinking` badge); fast active-1.5B MoE decode (~55–85 tok/s est.) helps absorb CoT length, though `8B` total weights occupy ~6.5 GB RAM.
- **Operational & pydantic-ai Readiness** `[Primary]`: **`Out-of-the-box`** (`Minor config` — test Pythonic `<|tool_call_start|>` parsing in Ollama `/v1`) via `OllamaModel` (`'ollama:lfm2.5:8b-a1b-q4_K_M'`).

## Benchmarks & Evidence

Liquid AI reported benchmarks (`LiquidAI/LFM2.5-8B-A1B` README, `[Primary]`):

- `IFEval`: `91.84` | `IFBench`: `56.47` | `Multi-IF`: `79.93`
- `BFCL v3`: `64.79` | `BFCL v4`: `49.73` (`48.50` in v1 table)
- `Tau² Telecom / Retail`: `88.07 / 39.82`
- `MATH500`: `88.76` | `AIME25`: `42.53` | `AIME26`: `50.00`
- `AA-Omniscience`: Non-hallucination `63.47%`, Accuracy `8.67%`, Index `-24.70`

## Caveats & Open Questions

- **Superseded by `lfm3:1.7b`**: Newer `lfm3:1.7b` beats `lfm2.5:8b` on `BFCL v4`, `IFBench`, `IFEval`, and non-hallucination while using less than half the RAM.
- **License Threshold ($10M Revenue Cap)**: `LFM Open License v1.0` restricts commercial use to entities under $10M annual revenue.
- **Pythonic Tool Format**: Verify Ollama's `/v1/chat/completions` tool parser on `lfm2.5:8b` before relying on `ToolOutput`.

## Sources

- `[2026-11-26]` [Ollama Library: lfm2.5](https://ollama.com/library/lfm2.5) & [lfm2.5/tags](https://ollama.com/library/lfm2.5/tags) — `8b` (1.5B active) and `24b` (2B active) tags, capabilities (`tools`, `thinking`).
- `[2026-11-26]` [Hugging Face: LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/raw/main/README.md) — MoE specs (`8.3B / 1.5B`), `IFEval`/`IFBench`/`BFCL`/`Tau2` benchmarks, Pythonic tool-call format, LFM Open License v1.0.
