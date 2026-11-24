import ast
from datetime import UTC, datetime
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
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


async def test_tool_registered_and_override():
    agent = build_shell_agent('anthropic:claude-sonnet-5-5')
    model = TestModel(call_tools=[])
    with agent.override(model=model):
        result = await agent.run('hi', deps=DEPS)
    assert result.output == 'success (no tool calls)'
    assert model.last_model_request_parameters is not None
    assert [t.name for t in model.last_model_request_parameters.function_tools] == ['session_info', 'read_pyai_docs']


async def test_session_info_runs():
    agent = build_shell_agent(TestModel(call_tools=['session_info']))
    result = await agent.run('hi', deps=DEPS)
    assert 'pax_x' in result.output


async def test_function_model_receives_deps():
    seen: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart('session_info', {})])
        seen.append(str(returns[0].content))
        return ModelResponse(parts=[TextPart('done')])

    agent = build_shell_agent(FunctionModel(respond))
    result = await agent.run('hi', deps=DEPS)
    assert result.output == 'done'
    assert 'pax_x' in seen[0] and "'logfire_remote': False" in seen[0]


async def test_thinking_capability_does_not_break_run():
    agent = build_shell_agent(TestModel(), thinking='high')
    assert 'pax_x' in (await agent.run('hi', deps=DEPS)).output


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
