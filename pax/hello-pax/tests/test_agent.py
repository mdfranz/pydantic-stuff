import ast
from datetime import UTC, datetime
from pathlib import Path

from pydantic_ai.models.test import TestModel

from hello_pax.agent_adapter.agent import ShellDeps, build_shell_agent

DEPS = ShellDeps(
    session_id='pax_x',
    model_label='test',
    started_at=datetime(2026, 1, 1, tzinfo=UTC),
    logfire_remote=False,
    log_path=None,
)


def test_build_without_keys():
    agent = build_shell_agent('anthropic:claude-sonnet-5-5', model_settings={'temperature': 0.1})
    assert agent.name == 'hello_pax_shell_agent'
    assert 'session_info' in agent._function_toolset.tools  # pyright: ignore[reportPrivateUsage]


async def test_session_info_runs():
    agent = build_shell_agent(TestModel(call_tools=['session_info']))
    result = await agent.run('hi', deps=DEPS)
    assert 'pax_x' in result.output


def test_import_boundary():
    root = Path(__file__).parent.parent / 'src' / 'hello_pax'
    for path in root.rglob('*.py'):
        if 'agent_adapter' in path.relative_to(root).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or '']
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(n.split('.')[0] == 'pydantic_clai2' for n in names), path
