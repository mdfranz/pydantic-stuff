import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.models.test import TestModel

from hello_pax.agent_adapter.agent import build_shell_agent
from hello_pax.telemetry import (
    JsonlSpanExporter,
    configure_telemetry,
    keep_content,
    resolve_log_path,
    resolve_logfire_content,
    resolve_remote_logfire,
)
from tests.test_agent import DEPS


def test_remote():
    assert resolve_remote_logfire(False, {})
    assert not resolve_remote_logfire(True, {})
    for v in ('0', 'false', 'off', 'no'):
        assert not resolve_remote_logfire(False, {'HELLO_PAX_LOGFIRE': v})
    assert resolve_remote_logfire(False, {'HELLO_PAX_LOGFIRE': '1'})


def test_content():
    assert not resolve_logfire_content(False, {})
    assert resolve_logfire_content(True, {})
    assert resolve_logfire_content(False, {'HELLO_PAX_LOGFIRE_CONTENT': 'yes'})


def test_log_path(tmp_path: Path):
    assert resolve_log_path(None, {}) is None
    p = resolve_log_path('auto', {'HELLO_PAX_STATE_DIR': str(tmp_path)})
    assert p is not None and p.parent == tmp_path / 'logs' and p.name.startswith('hello-pax-')
    assert not (tmp_path / 'logs').exists()
    assert resolve_log_path('/x/y.jsonl', {}) == Path('/x/y.jsonl')
    assert resolve_log_path('off', {'HELLO_PAX_LOG_FILE': '/z'}) is None
    assert resolve_log_path('/a', {'HELLO_PAX_LOG_FILE': '/z'}) == Path('/a')
    assert resolve_log_path(None, {'HELLO_PAX_LOG_FILE': '/z'}) == Path('/z')


def test_exporter_lazy(tmp_path: Path):
    target = tmp_path / 'd' / 'x.jsonl'
    JsonlSpanExporter(target).shutdown()
    assert not target.parent.exists()


def test_keep_content():
    assert keep_content(SimpleNamespace(path=('attributes', 'gen_ai.tool.call.result'), value='v')) == 'v'
    assert keep_content(SimpleNamespace(path=('attributes', 'note'), value='v')) is None


def test_real_spans(tmp_path: Path):
    import logfire

    target = tmp_path / 'logs' / 'x.jsonl'
    configure_telemetry(log_path=target, remote=False, service_version='t')
    agent = build_shell_agent(TestModel(call_tools=['session_info']))
    agent.run_sync('hi', deps=DEPS)
    logfire.force_flush()
    records = [json.loads(line) for line in target.read_text().splitlines()]
    assert any(r['attributes'].get('gen_ai.tool.name') == 'session_info' for r in records)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700


@pytest.fixture(autouse=True)
def _no_remote(monkeypatch: pytest.MonkeyPatch):  # pyright: ignore[reportUnusedFunction]
    monkeypatch.delenv('LOGFIRE_TOKEN', raising=False)
