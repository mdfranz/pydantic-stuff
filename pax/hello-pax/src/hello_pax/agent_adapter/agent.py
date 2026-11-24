from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

DEFAULT_INSTRUCTIONS = (
    'You are a concise, helpful assistant running in the hello-pax terminal shell. '
    'You have no filesystem, shell, or network tools. '
    'Use the session_info tool when asked about this session (id, model, start time, telemetry).'
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ShellDeps:
    session_id: str
    model_label: str
    started_at: datetime
    logfire_remote: bool
    log_path: Path | None


def session_info(ctx: RunContext[ShellDeps]) -> dict[str, str | bool | None]:
    """Return read-only metadata about the current hello-pax session."""
    deps = ctx.deps
    return {
        'session_id': deps.session_id,
        'model': deps.model_label,
        'started_at': deps.started_at.isoformat(),
        'logfire_remote': deps.logfire_remote,
        'log_path': None if deps.log_path is None else str(deps.log_path),
    }


def build_shell_agent(
    model: Model | str, *, instructions: str | None = None, model_settings: ModelSettings | None = None
) -> Agent[ShellDeps, str]:
    return Agent(
        model,
        name='hello_pax_shell_agent',
        deps_type=ShellDeps,
        output_type=str,
        instructions=instructions or DEFAULT_INSTRUCTIONS,
        tools=[session_info],
        model_settings=model_settings,
        defer_model_check=True,
    )
