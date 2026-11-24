from datetime import UTC, datetime
from pathlib import Path

import pytest

from hello_pax.agent_adapter.status_plugin import StatusControl, StatusPlugin, current_status


def make(remote: bool = True, log: Path | None = None) -> StatusControl:
    return StatusControl(
        session_id='pax_x', model_label='test', started_at=datetime(2026, 1, 1, tzinfo=UTC),
        logfire_remote=remote, log_path=log, request_limit=200,
    )  # fmt: skip


async def test_handle():
    c = make()
    assert 'pax_x' in await c.handle([])
    assert 'logfire' in await c.handle(['telemetry'])
    assert 'model: test' in await c.handle(['model'])
    assert (await c.handle(['--help'])).startswith('usage')
    assert await c.handle(['-h']) == await c.handle(['--help'])
    assert 'unknown' in await c.handle(['zzz'])


def test_completions():
    c = make()
    assert 'telemetry' in c.completions([])
    assert c.completions(['telemetry', 'x']) == ()


@pytest.mark.parametrize(
    ('remote', 'log', 'text'),
    [
        (True, Path('a'), 'hello-pax · logfire+jsonl'),
        (True, None, 'hello-pax · logfire'),
        (False, Path('a'), 'hello-pax · jsonl'),
        (False, None, 'hello-pax · telemetry:off'),
    ],
)
def test_segment(remote: bool, log: Path | None, text: str):
    assert make(remote, log).status_segment() == text


def test_plugin_binding():
    plugin = StatusPlugin.__new__(StatusPlugin)
    assert plugin.get_capabilities() == ()
    assert plugin.get_commands() == () and plugin.get_status_segments() == ()
    c = make()
    token = current_status.set(c)
    try:
        assert plugin.get_commands()[0].name == 'status'
        assert plugin.get_status_segments() == (c.status_segment,)
    finally:
        current_status.reset(token)
