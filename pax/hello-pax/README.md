# hello-pax

A minimal, isolated [`pydantic-clai2`](https://pypi.org/project/pydantic-clai2/) terminal shell hosting a
[Pydantic AI](https://ai.pydantic.dev) agent, with Logfire-first telemetry. See [PLAN.md](PLAN.md) for the design.

## Quick start
```sh
uv sync
./hello-pax.sh                    # default model google:gemini-3.8-flash (needs GEMINI_API_KEY)
./hello-pax.sh -m anthropic:claude-sonnet-5-5 --thinking high
./hello-pax.sh -j                 # also write a local 0600 JSONL span log
./hello-pax.sh -h                 # all options
```
Python >= 3.14 and [`uv`](https://docs.astral.sh/uv/) are required.

## Keys (environment variables)
| Variable | Used for |
| --- | --- |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `google:` models and `/search google` |
| `ANTHROPIC_API_KEY` | `anthropic:` models and `/search anthropic` |
| `OPENAI_API_KEY` | `openai:` models and `/search openai` |
| `OPENROUTER_API_KEY` | `openrouter:` models |
| `LOGFIRE_TOKEN` | sending traces to Logfire (without it, nothing is sent and the shell says so) |
| `PYDANTIC_AI_GATEWAY_API_KEY` | `gateway/...` models, or a bare model with `--route` |

## Options and environment
| Flag | Env | Effect |
| --- | --- | --- |
| `-m, --model` | `HELLO_PAX_MODEL` | model string (default `google:gemini-3.8-flash`) |
| `-r, --route` | `HELLO_PAX_GATEWAY_ROUTE` | Gateway route for a bare model name |
| `--thinking LEVEL` | | `minimal`..`max` |
| `-j, --jsonl` / `--log-file PATH\|auto\|off` | `HELLO_PAX_LOG_FILE` | local JSONL span log |
| `--no-logfire` | `HELLO_PAX_LOGFIRE=0` | do not send spans to Logfire |
| `--content` / `--logfire-content` | `HELLO_PAX_LOGFIRE_CONTENT=1` | keep GenAI content out of Logfire's scrubber |
| `--ephemeral` | | temporary CLAI2 settings store |

State lives under `$HELLO_PAX_STATE_DIR`, else `$XDG_STATE_HOME/hello-pax`, else `~/.local/state/hello-pax`
(directories `0700`, JSONL files `0600`).

## In the shell
| Command | What it does |
| --- | --- |
| `/status [telemetry\|model]` | session, model, and telemetry summary (operator-only) |
| `/search <google\|anthropic\|openai> <query>` | native web search, shown to you and not added to the conversation |
| `/refine [focus]` | have the model review and revise its last answer |
| `/compact` | summarise older messages; the status row shows a context gauge |

The model has one tool, `session_info`, plus CLAI2's inline `ask_user` questions. It has no filesystem, shell, or
network tools. CLAI2's own `!COMMAND` and `/monty` are operator-side shell features and are left enabled.

## Development
```sh
uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest
```
Tests are hermetic: no network, no credentials, no TUI. Only `src/hello_pax/agent_adapter/` may import
`pydantic_clai2` (enforced by a test). Search model overrides: `HELLO_PAX_SEARCH_{GOOGLE,ANTHROPIC,OPENAI}_MODEL`.
