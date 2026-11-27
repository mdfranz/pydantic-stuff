# AGENTS.md — Small Ollama Decision-Model Research Guide

## Mission & Scope

This workspace guides URL-driven web research, evaluation, and side-by-side comparison of **small Ollama-runnable decision models** running locally on an **Apple Mac Mini M4 (24GB unified memory)**.

**Target workloads ("decision models")**:
- **Classification & triage**: routing, intent detection, risk/priority scoring, labeling.
- **Structured output**: strict JSON generation, enum selection, schema adherence.
- **Tool & function calling**: single- and multi-step tool selection and argument accuracy.
- **Policy & rule evaluation**: deterministic compliance checks, rubrics, guardrail decisions.
- **Constrained reasoning**: fast, reliable judgment under tight local memory and latency budgets.

**Target stack (`pydantic-ai`)**: Models will be integrated via **`pydantic-ai`** (using Ollama's OpenAI-compatible endpoint). Research and document `pydantic-ai` compatibility, **do not treat out-of-the-box support as a hard exclusion criterion**—when gaps exist (e.g., missing tool-call support, thinking-tag parsing issues, needing `NativeOutput`/`PromptedOutput`, or a custom `ModelProfile`/Modelfile), record the gap and the concrete workaround so it can be addressed directly.

Out of scope unless explicitly requested: large frontier/cloud-only models, open-ended creative writing benchmarks, and models that require >18GB unified memory just to load.

---

## Target Hardware Profile (Mac Mini M4 — 24GB Unified Memory)

- **Compute & Memory**: Apple M4 (base M4 ~120 GB/s unified memory bandwidth; note M4 Pro is ~273 GB/s if comparing external M4 Pro reports), 24GB unified memory, macOS Metal backend via Ollama.
- **Practical Memory Budget**:
  - **~16–18 GB max usable** for model weights + KV cache + context + Metal runtime before macOS memory pressure and swap degrade performance.
  - **≤ ~12 GB total footprint** is the comfortable daily sweet spot (leaves headroom for macOS, apps, and concurrent tools).

### Fit Tiers

| Fit Tier | Est. Total RAM Footprint (Weights + Working Context) | Typical Profile | Notes |
| --- | --- | --- | --- |
| `Comfortable` | ≤ ~12 GB | 1B–14B dense at `Q4_K_M`, or compact MoE | Smooth daily driver with 8k–32k context and OS headroom. |
| `Tight` | ~12–17 GB | ~14B at `Q6_K`/`Q8_0`, ~20B–24B at `Q4_K_M`, or mid-size MoE | Usable with tuned context length, `OLLAMA_FLASH_ATTENTION=1`, and `OLLAMA_KV_CACHE_TYPE=q8_0`. |
| `Too Large` | > ~17–18 GB | 27B–32B+ at `Q4_K_M` or larger | Risks swap thrashing and unacceptable latency; document and exclude from primary shortlists. |

### Bandwidth-Based Decode Speed Sanity Check

- **Theoretical upper bound**: `~120 GB/s ÷ active_model_size_GB` (tok/s).
- **Realistic sustained decode**: typically **~65–80% of theoretical** on M4 Metal, before accounting for prompt-eval (TTFT) or large-context KV-cache overhead.
- **MoE rule**: decode speed is governed by **active** parameter weights per token, while RAM fit is governed by **total** loaded weights.

---

## Repository Layout (Created On Demand)

Do **not** pre-create empty placeholder files or folders. Create them lazily as research progresses:

- `AGENTS.md` — this research & evaluation guide.
- `sources.md` — chronological log of user-provided URLs and discovered references (date, URL, source type, models mentioned, key takeaways).
- `models/<model-slug>.md` — per-model research dossier (lowercase kebab-case slug, e.g., `qwen3-8b.md`).
- `comparisons/matrix.md` — master side-by-side comparison table, M4 fit ratings, and task-based shortlists.
- `benchmarks/` — optional local Ollama test prompts, scripts, and measurements run directly on the Mac Mini M4.

---

## URL-Driven Research Workflow

When the user provides one or more URLs (or asks to investigate a model/article/thread):

1. **Ingest URLs**: Fetch/search content from user-supplied URLs and extract candidate models, exact Ollama tags, performance claims, and referenced benchmarks.
2. **Verify Against Primary Sources**: Cross-check Ollama library tags/sizes/quantizations, Hugging Face model cards, official technical reports/release blogs, and reputable independent benchmarks.
3. **Assess Mac Mini M4 (24GB) Fit**: Record disk/RAM footprint, estimate KV-cache overhead at realistic context windows (8k / 16k / 32k), assign a Fit Tier (`Comfortable` / `Tight` / `Too Large`), and compute the M4 bandwidth sanity check.
4. **Evaluate Decision-Model Capabilities**: Score/assess against the criteria below.
5. **Persist & Update Docs**:
   - Append entries to `sources.md`.
   - Create or update `models/<model-slug>.md`.
   - Refresh `comparisons/matrix.md` in place.
6. **Report Back Concisely**: Summarize key findings, compare against current shortlisted candidates, flag gaps/caveats, and suggest follow-up URLs or local M4 verification tests.

---

## Decision-Model Evaluation Criteria

Assess every candidate across these six dimensions:

1. **Decision Quality & Determinism**: Consistency at low temperature (`0.0–0.2`), multi-criteria judgment, ambiguity handling, and abstention/refusal calibration.
2. **Structured Output & Schema Adherence**: Strict JSON reliability, enum compliance, nested schema accuracy, compatibility with Ollama structured outputs (`format` / JSON schema), `pydantic-ai` output modes (`ToolOutput`, `NativeOutput`, `PromptedOutput`), and Pydantic validation-retry overhead on small models.
3. **Tool & Function Calling**: Native Ollama tool-calling support (including Ollama's OpenAI-compatible `/v1` tool-calling behavior as consumed by `pydantic-ai`), tool selection accuracy, argument precision, and multi-step call reliability.
4. **Latency & Thinking Overhead**: TTFT and decode tok/s on M4; whether reasoning/thinking mode is toggleable (e.g., `/no_think` or parameter flag) vs. always-on chain-of-thought that inflates decision latency.
5. **Memory & Context Efficiency**: Recommended quantization (`Q4_K_M`, `Q5_K_M`, `Q8_0`), quality degradation under 4-bit quant, and long-context retention within the 24GB budget.
6. **Operational & `pydantic-ai` Readiness**: Official `ollama.com/library` availability vs. community GGUF, Modelfile/chat-template maturity, license, known bugs/regressions, `pydantic-ai` status (`Out-of-the-box` / `Minor config` / `Needs workaround`), thinking-tag handling, preferred `pydantic-ai` output mode, and actionable notes on any Modelfile/template/`ModelProfile` patch needed if not supported out of the box (never exclude a strong model solely for an integration gap).

---

## Standard Templates

### Per-Model Dossier (`models/<model-slug>.md`)

```markdown
# <Model Name>

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `<exact-tag>` (e.g., `model:8b-q4_K_M`) |
| **Parameters (Total / Active)** | `<e.g., 8B / 8B>` |
| **Architecture** | `<Dense / MoE>` |
| **Context Window (Max / Practical M4)** | `<e.g., 128k / 16k>` |
| **Recommended Quantization** | `<e.g., Q4_K_M>` |
| **License** | `<e.g., Apache-2.0>` |
| **pydantic-ai Readiness** | `<Out-of-the-box / Minor config / Needs workaround / TBD>` |
| **Status** | `<Shortlisted / Candidate / Excluded>` |
| **Last Updated** | `<YYYY-MM-DD>` |

## Mac Mini M4 (24GB) Fit

- **Weight Size (Disk / Loaded)**: `TBD`
- **Est. Total RAM @ 8k / 16k / 32k**: `TBD`
- **Fit Tier**: `<Comfortable | Tight | Too Large>`
- **Est. M4 Decode Speed**: `TBD` (sanity check: `~120 GB/s ÷ active_GB × 0.65–0.80`)
- **Recommended Ollama Env / Settings**: e.g., `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` if needed.

## Decision-Model Assessment

- **Decision Quality & Determinism**: `TBD`
- **Structured Output & JSON Schema**: `TBD`
- **Tool & Function Calling**: `TBD`
- **Thinking / Latency Tradeoff**: `TBD`
- **Quantization & Context Notes**: `TBD`
- **Operational & pydantic-ai Readiness**: `TBD` (preferred output mode: `ToolOutput`/`NativeOutput`/`PromptedOutput`; gaps & DIY workarounds if any)

## Benchmarks & Evidence

- Summarize relevant benchmark numbers and local measurements with evidence labels (`[Primary]`, `[Benchmark]`, `[Community]`, `[Estimated]`, `[Measured]`).

## Caveats & Open Questions

- Quantization sensitivity, template quirks, known Ollama issues, or missing measurements.

## Sources

- `[YYYY-MM-DD]` [Source Title](https://example.com) — key contribution.
```

### Master Comparison Matrix (`comparisons/matrix.md`)

Maintain a primary table using this exact column schema, followed by short task-based shortlists (e.g., Fastest Router, Best Strict JSON, Best Tool Caller, Best Heavy Reasoner within 24GB):

```markdown
| Model | Ollama Tag | Params (Total/Active) | Quant | Size / Est. RAM | M4 24GB Fit | Tools | JSON | Thinking | pydantic-ai | Est. M4 Speed | Best For | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
```

---

## Evidence, Citations & Integrity Rules

- **Exact Identifiers**: Always record exact Ollama tags (never vague names like "Llama 8B"), exact source URLs, and access dates (`YYYY-MM-DD`).
- **Evidence Labels**: Tag claims consistently:
  - `[Primary]` — official model card, Ollama library page, technical report, or vendor release notes.
  - `[Benchmark]` — published eval suite or reproducible leaderboard result.
  - `[Community]` — blog posts, Reddit/HN/GitHub discussions, user reports.
  - `[Estimated]` — derived hardware math (e.g., M4 bandwidth/RAM estimates).
  - `[Measured]` — directly observed locally on the target Mac Mini M4.
- **No Fabrication**: Never invent benchmark scores, parameter counts, model sizes, or Ollama tags. Mark unknown fields `TBD`.
- **Call Out Caveats Explicitly**: Always flag quantization degradation risks, benchmark contamination/gaming, M4 vs. M4 Pro/Max hardware mismatches in cited reports, and thinking-token latency costs.
- **Edit In Place**: Update existing entries in `sources.md`, `models/<model-slug>.md`, and `comparisons/matrix.md` rather than duplicating files. Keep summaries concise and scannable.
