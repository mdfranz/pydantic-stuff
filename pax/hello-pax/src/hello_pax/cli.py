import argparse
import os
import sys
from collections.abc import Sequence
from functools import partial
from typing import cast

import anyio
from pydantic_ai.settings import ThinkingLevel

from . import package_version
from .agent_adapter import DEFAULT_REQUEST_LIMIT, run_shell
from .errors import HelloPaxError
from .models_config import resolve_model
from .telemetry import (
    announce,
    configure_telemetry,
    resolve_log_path,
    resolve_logfire_content,
    resolve_remote_logfire,
)

EXIT_OK = 0
EXIT_FATAL = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130
DEFAULT_MODEL = 'google:gemini-3.8-flash'
THINKING_LEVELS = ('minimal', 'low', 'medium', 'high', 'xhigh', 'max')


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='hello-pax', description='Minimal pydantic-clai2 shell with Logfire telemetry')
    p.add_argument(
        '-m',
        '--model',
        help='pydantic-ai model string (or bare name with --gateway-route); $HELLO_PAX_MODEL (default: google:gemini-3.8-flash)',
    )
    p.add_argument('-r', '--gateway-route', help='Pydantic AI Gateway route for a bare --model')
    p.add_argument('--thinking', choices=THINKING_LEVELS, help='reasoning effort')
    p.add_argument('--instructions', help='custom system prompt')
    p.add_argument(
        '--request-limit', type=_positive_int, default=DEFAULT_REQUEST_LIMIT, help='max model requests per prompt'
    )
    p.add_argument('--ephemeral', action='store_true', help='use a temporary CLAI2 settings store')
    p.add_argument('--no-logfire', action='store_true', help='do not send spans to Logfire ($HELLO_PAX_LOGFIRE=0)')
    p.add_argument(
        '--logfire-content',
        action='store_true',
        help='keep GenAI content out of the scrubber ($HELLO_PAX_LOGFIRE_CONTENT=1)',
    )
    p.add_argument('--log-file', help='local JSONL span log: PATH, auto, or off ($HELLO_PAX_LOG_FILE)')
    p.add_argument('--version', action='version', version=f'hello-pax {package_version()}')
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_OK
    model_arg = args.model or os.environ.get('HELLO_PAX_MODEL') or DEFAULT_MODEL
    try:
        model = resolve_model(model_arg, gateway_route=args.gateway_route)
        remote = resolve_remote_logfire(args.no_logfire)
        log_path = resolve_log_path(args.log_file)
        content = resolve_logfire_content(args.logfire_content)
        if remote or log_path is not None:
            active = configure_telemetry(
                log_path=log_path, remote=remote, service_version=package_version(), content=content
            )
            announce(log_path, remote=active, requested=remote)
            remote = active
        anyio.run(
            partial(
                run_shell,
                model,
                instructions=args.instructions,
                request_limit=args.request_limit,
                thinking=cast('ThinkingLevel | None', args.thinking),
                logfire_remote=remote,
                log_path=log_path,
                ephemeral=args.ephemeral,
            )
        )
    except HelloPaxError as exc:
        print(f'hello-pax: {exc.rule_code or "error"}: {exc}', file=sys.stderr)
        return EXIT_FATAL
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    return EXIT_OK
