# Ornith 1.5 9B (`ornith-1.5:9b`)

## Metadata

| Field | Value |
| --- | --- |
| **Ollama Tag(s)** | `ornith-1.5:9b` (`e5df7dcdd8a2`, `6.6 GB` GGUF) *(larger `35b`, `122b`, `397b` tags are `Too Large` for 24GB M4)* |
| **Parameters (Total / Active)** | `9B` / `9B` (dense) |
| **Architecture** | `Qwen3.5-9B` hybrid (`Qwen3_5ForConditionalGeneration`: 32 layers = 24 Gated DeltaNet `linear_attention` + 8 `full_attention`, `num_key_value_heads=4`, `head_dim=256`, multimodal vision) |
| **Context Window (Max / Practical M4)** | `256K` (`262,144`) max *(1M via YaRN)* / `8k–32k` practical M4 |
| **Recommended Quantization** | `ornith-1.5:9b` (`6.6 GB`, only 9B tag published on Ollama) |
| **License** | **MIT** |
| **pydantic-ai Readiness** | `Out-of-the-box` (`Minor config`: verify `<think>` reasoning extraction & `qwen3_xml` tool parser in Ollama) |
| **Status** | `Shortlisted` (**Best Heavy Reasoner / System 2 `<10B`** on 24GB M4) |
| **Last Updated** | `2026-11-26` |

## Ollama Tags & Mac Mini M4 (24GB) Fit

| Ollama Tag | Hash | Disk Size | Input | M4 24GB Fit | Notes |
| --- | --- | ---: | --- | --- | --- |
| `ornith-1.5:9b` | `e5df7dcdd8a2` | **6.6 GB** | Text, Image | `Comfortable` | Only `<10B` tag on Ollama (no `MLX` or `q8_0` 9B tag published) |
| `ornith-1.5:35b` | `9f3b89b25219` | 23 GB | Text, Image | `Too Large` | `35B-A3B` MoE; weights exceed 24GB budget |
| `ornith-1.5:122b` (`nvfp4`/`q4_K_M`) | `ea3e29aebeb8` | 73–77 GB | Text, Image | `Too Large` | Exceeds 24GB |
| `ornith-1.5:397b` | `337c6db6be12` | 242 GB | Text, Image | `Too Large` | Exceeds 24GB |

- **Weight Size (Disk / Loaded)** `[Primary]`: `6.6 GB` (`ornith-1.5:9b`).
- **KV Cache Math** `[Estimated]`: `Qwen3.5-9B` 3:1 hybrid attention (~32 KB/token for fp16 KV cache: ~256 MB @ 8k, ~512 MB @ 16k, ~1.0 GB @ 32k).
- **Est. Total RAM @ 8k / 16k / 32k** `[Estimated]`: **~7.8 GB** @ 8k / **~8.1 GB** @ 16k / **~8.8 GB** @ 32k (`Comfortable` — leaves ~15+ GB free; can stay co-resident alongside `laya` or `d3:4b`).
- **Fit Tier**: `Comfortable` (`ornith-1.5:9b`); `Too Large` (`35b`, `122b`, `397b`).
- **Est. M4 Decode Speed** `[Estimated]` (`~120 GB/s ÷ 6.6 GB × 0.65–0.80`): **~11.8–14.5 tok/s**.
- **Recommended Ollama Env / Settings**: `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` for long contexts. HF recommends `temperature=0.6, top_p=0.95, top_k=20` for precise coding and `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5` for general tasks `[Primary]`.

## Decision-Model Assessment

- **Decision Quality & Determinism** `[Primary]`: Exceptional reasoning and agentic performance for a 9B dense model (RL self-improvement loop over task generation, scaffolds, and rollouts, released Aug 19, 2026). Scores **`86.4` on GPQA Diamond** (beating `Qwen3.6-35B-A3B` `86.0` and `Gemma-4-31B` `84.3`), **`30.5` on HLE with tools**, and **`70.6` on SWE-bench Verified** (vs. `Qwen3.5-9B` `53.2`). Note that benchmark defaults use `temp=0.6–1.0` with thinking enabled; determinism at `temp=0.0–0.2` for classification is unverified.
- **Structured Output & JSON Schema** `[Primary]`: Because `ornith-1.5:9b` uses GGUF (`6.6 GB`), self-hosted Ollama enforces `json_schema` via `llama.cpp` grammar-constrained decoding for `pydantic-ai` `NativeOutput`.
- **Tool & Function Calling** `[Primary]`: Strong agentic tool scores (`Toolathlon-Verified 41.2` vs. `MiMo-V2.6-Distill-9B` `35.2` and `Qwen3.5-9B` `29.6`; `MCP-Atlas 54.2`; `ClawEval 66.5`). Emits `<tool_call>` blocks (`qwen3_xml` / `qwen3_coder` parsers on vLLM/SGLang).
- **Thinking / Latency Tradeoff** `[Primary]`: **Always opens with `<think> ... </think>` by default** (`Ornith-1.5-9B is a reasoning model`). HF card documents reasoning-mode evaluation (`5-run averages` with long context/token budgets) and does not advertise a no-think mode. At ~12–14 tok/s on M4, long `<think>` traces make it **unsuitable as a fast router**, and **ideal as a System 2 escalation target (`ModelFallback`)**.
- **Quantization & Context Notes** `[Primary]`: Only a single `6.6 GB` (`~Q4_K_M`) tag is available on Ollama (no `Q8_0` or `MLX` tags yet), so test locally for any 4-bit quantization loss on complex schemas. 256K native context (`Text, Image`).
- **Operational & pydantic-ai Readiness** `[Primary]`: **`Out-of-the-box`** (`Minor config`) via `OllamaModel` (`'ollama:ornith-1.5:9b'`) with `OLLAMA_BASE_URL='http://localhost:11434/v1'`.
  - **Preferred output mode**: `NativeOutput` (grammar-constrained JSON) or `ToolOutput`.
  - **Gaps to verify locally**: Confirm Ollama's chat template cleanly parses `<think>` into `ThinkingPart` and `<tool_call>` into OpenAI `tool_calls` (HF notes a modified Qwen chat template was needed in vLLM/Harbor).

## Benchmarks & Evidence

Ornith AI reported 5-run average benchmarks (`ornith-ai/Ornith-1.5-9B` README, thinking mode, `[Primary]`):

| Benchmark | `Ornith-1.5-9B` | `Ornith-1.0-9B` | `Qwen3.5-9B` | `Qwen3.6-35B-A3B` | `Gemma-4-31B` |
| --- | ---: | ---: | ---: | ---: | ---: |
| **GPQA Diamond** | **86.4** | 82.5 | 81.7 | 86.0 | 84.3 |
| **HLE (no tools / with tools)** | **20.2 / 30.5** | 16.8 / 26.4 | 14.7 / 24.5 | 21.4 / 28.9 | 19.5 / 26.5 |
| **Toolathlon-Verified** | **41.2** | 33.4 | 29.6 | 41.7 | 52.8 |
| **MCP-Atlas** | **54.2** | 49.4 | 46.8 | 62.8 | 55.0 |
| **ClawEval** | **66.5** | 63.1 | 53.2 | 68.7 | 48.5 |
| **WideSearch / BrowseComp** | **59.5 / 56.4** | 55.8 / 44.8 | 53.6 / 41.5 | 60.1 / 62.0 | 54.2 / — |
| **SWE-bench Verified / Pro** | **70.6 / 47.5** | 69.4 / 42.9 | 53.2 / 31.3 | 73.4 / 49.5 | 52.0 / 35.7 |
| **Terminal-Bench 2.1 (Terminus-2 / Claude Code)** | **46.2 / 47.0** | 43.1 / 40.6 | 21.3 / 18.9 | 52.5 / 49.2 | 42.1 / — |

*(Caveat: Ornith's evaluation harnesses/sampling differ from Xiaomi's `MiMo-V2.6` table — e.g., Ornith evaluates SWE-bench with OpenHands `temp=1.0`, Toolathlon at 128K, and Terminal-Bench with Terminus-2/Claude Code, so cross-card percentages are directional rather than strictly apples-to-apples.)*

## Caveats & Open Questions

- **Heavy Thinking Latency**: Default `<think>` reasoning at ~12–14 tok/s on base M4 is too slow for high-frequency routing; use behind a System One router (`d3`, `kev`, `laya`).
- **Single Ollama Tag (`6.6 GB`)**: No `Q8_0` or `MLX` tags published on `ollama.com/library/ornith-1.5` yet.
- **No Local M4 Measurements Yet**: Ollama `<tool_call>` / `<think>` template parsing, `pydantic-ai` `NativeOutput` / `ToolOutput` reliability, and M4 tok/s are `TBD`.

## Sources

- `[2026-11-26]` [Ollama Library: ornith-1.5](https://ollama.com/library/ornith-1.5) & [ornith-1.5/tags](https://ollama.com/library/ornith-1.5/tags) — `ornith-1.5:9b` (`6.6 GB`, `Text, Image`, 256K ctx), capabilities (`vision`, `tools`, `thinking`), Aug 19, 2026 release.
- `[2026-11-26]` [Hugging Face: ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B/raw/main/README.md) & [config.json](https://huggingface.co/ornith-ai/Ornith-1.5-9B/raw/main/config.json) — MIT license, `Qwen3.5-9B` hybrid architecture, 5-run benchmark tables, `<think>` and `qwen3_xml` serving notes.
- `[2026-11-26]` [Ornith 1.5 Blog](https://ornith.ai/ornith_1_5.html) — task/scaffold/rollout RL loop methodology and eval settings.
