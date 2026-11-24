#!/usr/bin/env bash
# hello-pax.sh: open the hello-pax CLAI2 terminal shell (Logfire telemetry on by default).
#
#   ./hello-pax.sh                                   # default model google:gemini-3.8-flash, logfire on
#   ./hello-pax.sh -m anthropic:claude-sonnet-5-5 --thinking high
#   ./hello-pax.sh -j                                # also write a local 0600 JSONL span log
#
# Usage: ./hello-pax.sh [options] [-- extra hello-pax args]   (flags override the environment variables in [])
#   -m, --model NAME        model string [HELLO_PAX_MODEL, default google:gemini-3.8-flash]
#   -r, --route NAME        Pydantic AI Gateway route for a bare model name [HELLO_PAX_GATEWAY_ROUTE]
#       --thinking LEVEL    minimal|low|medium|high|xhigh|max
#   -j, --jsonl             write a local JSONL span log (--log-file auto) [HELLO_PAX_JSONL=1]
#       --no-logfire        do not send spans to Logfire [HELLO_PAX_LOGFIRE=0]
#       --content           keep GenAI prompt/reply/tool content out of the scrubber [HELLO_PAX_LOGFIRE_CONTENT=1]
#       --ephemeral         use a temporary CLAI2 settings store
#       --version           print the hello-pax version
#   -h, --help
# Keys are read from the environment by pydantic-ai / logfire: GEMINI_API_KEY (google:), ANTHROPIC_API_KEY
# (anthropic:), OPENROUTER_API_KEY (openrouter:), LOGFIRE_TOKEN (remote export), and PYDANTIC_AI_GATEWAY_API_KEY
# for a bare model with --route or a gateway/... model.
set -euo pipefail

usage() { sed -n '/^# Usage:/,/^#   -h/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"; }
MODEL=${HELLO_PAX_MODEL:-}
ROUTE=${HELLO_PAX_GATEWAY_ROUTE:-}
JSONL=${HELLO_PAX_JSONL:+1}
args=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model) MODEL=${2?$1 needs a value}; shift 2 ;;
    -r|--route) ROUTE=${2?$1 needs a value}; shift 2 ;;
    --thinking) args+=(--thinking "${2?$1 needs a value}"); shift 2 ;;
    -j|--jsonl) JSONL=1; shift ;;
    --no-logfire) args+=(--no-logfire); shift ;;
    --content) args+=(--logfire-content); shift ;;
    --ephemeral) args+=(--ephemeral); shift ;;
    --version) args+=(--version); shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; args+=("$@"); break ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)  # absolute, so it works from any directory
cd "$SCRIPT_DIR"

[[ -z $JSONL ]] || args+=(--log-file auto)
[[ -z $MODEL ]] || args+=(--model "$MODEL")
[[ -z $ROUTE ]] || args+=(--gateway-route "$ROUTE")
if [[ -n $ROUTE || $MODEL == gateway/* ]] && [[ -z ${PYDANTIC_AI_GATEWAY_API_KEY:-} ]]; then
  echo 'set PYDANTIC_AI_GATEWAY_API_KEY for Gateway models' >&2
  exit 1
fi
exec uv run --quiet hello-pax ${args[@]+"${args[@]}"}
