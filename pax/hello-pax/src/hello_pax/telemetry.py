import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from .paths import mkdir_private, state_dir

LOGFIRE_ENV = 'HELLO_PAX_LOGFIRE'
LOGFIRE_CONTENT_ENV = 'HELLO_PAX_LOGFIRE_CONTENT'
LOG_FILE_ENV = 'HELLO_PAX_LOG_FILE'
CONTENT_ATTRIBUTES = frozenset(
    {
        'gen_ai.input.messages',
        'gen_ai.output.messages',
        'gen_ai.system_instructions',
        'gen_ai.tool.call.arguments',
        'gen_ai.tool.call.result',
        'final_result',
    }
)
_OFF = {'0', 'false', 'off', 'no'}
_ON = {'1', 'true', 'on', 'yes'}


def resolve_remote_logfire(no_logfire: bool, env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    if no_logfire:
        return False
    return env.get(LOGFIRE_ENV, '').strip().lower() not in _OFF


def resolve_logfire_content(flag: bool, env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return flag or env.get(LOGFIRE_CONTENT_ENV, '').strip().lower() in _ON


def resolve_log_path(cli_value: str | None, env: Mapping[str, str] | None = None) -> Path | None:
    env = os.environ if env is None else env
    value = cli_value if cli_value is not None else env.get(LOG_FILE_ENV)
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in {'off', 'none', '0', 'false'}:
        return None
    if value.lower() == 'auto':
        stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
        return state_dir(env) / 'logs' / f'hello-pax-{stamp}-{os.getpid()}.jsonl'
    return Path(value).expanduser()


def span_record(span: ReadableSpan) -> dict[str, Any]:
    ctx = span.get_span_context()
    parent = span.parent

    def iso(ns: int | None) -> str | None:
        return None if ns is None else datetime.fromtimestamp(ns / 1e9, UTC).isoformat()

    return {
        'name': span.name,
        'trace_id': None if ctx is None else format(ctx.trace_id, '032x'),
        'span_id': None if ctx is None else format(ctx.span_id, '016x'),
        'parent_span_id': None if parent is None else format(parent.span_id, '016x'),
        'start': iso(span.start_time),
        'end': iso(span.end_time),
        'status': span.status.status_code.name,
        'status_description': span.status.description,
        'attributes': dict(span.attributes or {}),
        'events': [
            {'name': e.name, 'timestamp': iso(e.timestamp), 'attributes': dict(e.attributes or {})} for e in span.events
        ],
    }


class JsonlSpanExporter(SpanExporter):
    """Append spans as JSONL; the file (0600) and parent dirs (0700) are created on first export."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: IO[str] | None = None

    def _open(self) -> IO[str]:
        if self._file is None:
            mkdir_private(self._path.parent)
            fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            self._file = os.fdopen(fd, 'a', encoding='utf-8')
        return self._file

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            handle = self._open()
            for span in spans:
                handle.write(json.dumps(span_record(span), default=str) + '\n')
            handle.flush()
        except OSError, ValueError:
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def keep_content(match: Any) -> Any:
    """Scrub callback: keep GenAI content attributes, scrub everything else."""
    path = match.path
    if len(path) >= 2 and path[0] == 'attributes' and path[1] in CONTENT_ATTRIBUTES:
        return match.value
    return None


def configure_telemetry(*, log_path: Path | None, remote: bool, service_version: str, content: bool = False) -> bool:
    """Configure Logfire; returns whether spans will actually be sent to Logfire (needs a token or credentials)."""
    import logfire

    processors = [] if log_path is None else [SimpleSpanProcessor(JsonlSpanExporter(log_path))]
    logfire.configure(
        service_name='hello-pax',
        service_version=service_version,
        send_to_logfire='if-token-present' if remote else False,
        console=False,
        additional_span_processors=processors,
        scrubbing=logfire.ScrubbingOptions(callback=keep_content) if content else None,
    )
    logfire.instrument_pydantic_ai()
    return remote and logfire.DEFAULT_LOGFIRE_INSTANCE.config.token is not None


NO_TOKEN_HINT = (
    'Logfire is not configured, so nothing is sent: run `uvx logfire auth` or set LOGFIRE_TOKEN (or --no-logfire)'
)


def announce(log_path: Path | None, *, remote: bool, requested: bool = False) -> None:
    parts: list[str] = []
    if remote:
        parts.append('logfire')
    if log_path is not None:
        parts.append(f'jsonl {log_path}')
    if requested and not remote:
        print(f'hello-pax: {NO_TOKEN_HINT}', file=sys.stderr)
    if parts:
        print(f'hello-pax: telemetry -> {", ".join(parts)}', file=sys.stderr)
