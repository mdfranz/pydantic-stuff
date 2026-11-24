from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic_clai2.commands import Command  # pyright: ignore[reportMissingTypeStubs]
from pydantic_clai2.plugins import NoSettings, Plugin  # pyright: ignore[reportMissingTypeStubs]
from pydantic_clai2.ui.rendering.status import StatusSegment  # pyright: ignore[reportMissingTypeStubs]

STATUS_PLUGIN_ID = 'hello_pax_status'
STATUS_PLUGIN_FACTORY = 'hello_pax.agent_adapter.status_plugin:StatusPlugin'
SUBCOMMANDS = ('telemetry', 'model', '--help')
USAGE = 'usage: /status [telemetry|model|--help]'


@dataclass(frozen=True, slots=True)
class StatusControl:
    session_id: str
    model_label: str
    started_at: datetime
    logfire_remote: bool
    log_path: Path | None
    request_limit: int
    thinking: str | None = None

    def status_segment(self) -> str:
        if self.logfire_remote and self.log_path is not None:
            return 'hello-pax · logfire+jsonl'
        if self.logfire_remote:
            return 'hello-pax · logfire'
        if self.log_path is not None:
            return 'hello-pax · jsonl'
        return 'hello-pax · telemetry:off'

    def completions(self, args: list[str]) -> tuple[str, ...]:
        return SUBCOMMANDS if len(args) <= 1 else ()

    def _telemetry(self) -> str:
        return (
            f'logfire: {"on (if token present)" if self.logfire_remote else "off"}\n'
            f'jsonl: {self.log_path if self.log_path is not None else "off"}'
        )

    def _model(self) -> str:
        return f'model: {self.model_label}\nthinking: {self.thinking or "default"}\nrequest limit: {self.request_limit}'

    async def handle(self, args: list[str]) -> str:
        if not args:
            return f'session: {self.session_id}\nstarted: {self.started_at.isoformat()}\n{self._model()}\n{self._telemetry()}'
        match args[0]:
            case 'telemetry':
                return self._telemetry()
            case 'model':
                return self._model()
            case '--help' | '-h':
                return USAGE
            case _:
                return f'unknown argument {args[0]!r}\n{USAGE}'


current_status: ContextVar[StatusControl | None] = ContextVar('hello_pax_status_control', default=None)


class StatusPlugin(Plugin[NoSettings]):
    def get_capabilities(self) -> Sequence[Any]:
        return ()

    def get_commands(self) -> Sequence[Command]:
        control = current_status.get()
        if control is None:
            return ()
        return (
            Command(
                name='status',
                description='Show hello-pax session, model, and telemetry status',
                handler=control.handle,
                complete=control.completions,
            ),
        )

    def get_status_segments(self) -> Sequence[StatusSegment]:
        control = current_status.get()
        return () if control is None else (control.status_segment,)
