import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic_ai.models import Model
from pydantic_ai.settings import ThinkingLevel
from pydantic_ai.usage import UsageLimits
from pydantic_clai2 import chat  # pyright: ignore[reportMissingTypeStubs]
from pydantic_clai2.config import PluginSettings, Settings  # pyright: ignore[reportMissingTypeStubs]
from pydantic_clai2.config.settings_store import SettingsStore  # pyright: ignore[reportMissingTypeStubs]
from rich.console import Console

from ..models_config import model_display_name
from ..paths import mkdir_private, state_dir
from .agent import ShellDeps, build_shell_agent
from .search_plugin import SEARCH_PLUGIN_FACTORY, SEARCH_PLUGIN_ID, SearchControl, current_search
from .status_plugin import STATUS_PLUGIN_FACTORY, STATUS_PLUGIN_ID, StatusControl, current_status

CHAT_SETTINGS = Settings(model=None, code_mode=False, speculative_code_mode=False, session_namer=False)
DEFAULT_REQUEST_LIMIT = 200

# Stock CLAI2 plugins we opt into, explicitly. None touch the filesystem, a shell, or hosted-service credentials.
STOCK_PLUGINS: tuple[PluginSettings, ...] = (
    PluginSettings(id='refinement', factory='pydantic_clai2.builtin_plugins.refinement'),
    PluginSettings(id='compaction', factory='pydantic_clai2.builtin_plugins.compaction', settings={}),
    PluginSettings(id='ask_user', factory='pydantic_clai2.builtin_plugins.ask_user_menu'),
)


async def run_shell(
    model: Model | str,
    *,
    instructions: str | None = None,
    request_limit: int = DEFAULT_REQUEST_LIMIT,
    thinking: ThinkingLevel | None = None,
    logfire_remote: bool = True,
    log_path: Path | None = None,
    ephemeral: bool = False,
    state_root: Path | None = None,
    console: Console | None = None,
) -> str:
    session_id = f'pax_{uuid4().hex[:12]}'
    started_at = datetime.now(UTC)
    model_label = model_display_name(model)
    deps = ShellDeps(
        session_id=session_id,
        model_label=model_label,
        started_at=started_at,
        logfire_remote=logfire_remote,
        log_path=log_path,
    )
    control = StatusControl(
        session_id=session_id,
        model_label=model_label,
        started_at=started_at,
        logfire_remote=logfire_remote,
        log_path=log_path,
        request_limit=request_limit,
        thinking=None if thinking is None else str(thinking),
    )
    agent = build_shell_agent(model, instructions=instructions, thinking=thinking)
    if logfire_remote or log_path is not None:
        import logfire

        logfire.info('hello-pax session started', session_id=session_id, model=model_label)

    async def _chat(config_db: Path) -> None:
        await chat(
            agent,
            deps=deps,
            load_plugins=True,
            store=SettingsStore(config_db),
            builtin_plugins=(
                *STOCK_PLUGINS,
                PluginSettings(id=STATUS_PLUGIN_ID, factory=STATUS_PLUGIN_FACTORY),
                PluginSettings(id=SEARCH_PLUGIN_ID, factory=SEARCH_PLUGIN_FACTORY),
            ),
            console=console,
            settings=CHAT_SETTINGS,
            usage_limits=UsageLimits(request_limit=request_limit),
        )

    token = current_status.set(control)
    search_token = current_search.set(SearchControl())
    try:
        if ephemeral:
            with tempfile.TemporaryDirectory(prefix='hello-pax-clai2-') as tmp:
                await _chat(Path(tmp) / 'config.db')
        else:
            chat_dir = (state_root or state_dir()) / 'clai2'
            mkdir_private(chat_dir)
            await _chat(chat_dir / 'config.db')
    finally:
        current_search.reset(search_token)
        current_status.reset(token)
    return session_id
