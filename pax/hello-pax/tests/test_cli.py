import subprocess
import sys
from typing import Any

import pytest

from hello_pax import cli


def test_version(capsys: pytest.CaptureFixture[str]):
    assert cli.main(['--version']) == 0
    assert capsys.readouterr().out.strip() == 'hello-pax 0.1.0'


def test_module_version():
    out = subprocess.run([sys.executable, '-m', 'hello_pax', '--version'], check=False, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == 'hello-pax 0.1.0'


def test_default_model(monkeypatch: pytest.MonkeyPatch):
    seen: list[Any] = []

    async def fake_run(model: Any, **kw: Any) -> str:
        seen.append(model)
        return 'id'

    monkeypatch.setattr(cli, 'run_shell', fake_run)
    monkeypatch.delenv('HELLO_PAX_MODEL', raising=False)
    assert cli.main(['--no-logfire']) == 0
    assert seen == ['google:gemini-3.8-flash']


def test_bad_model(capsys: pytest.CaptureFixture[str]):
    assert cli.main(['-m', 'bare']) == 1
    assert 'hello-pax: model-config:' in capsys.readouterr().err


def test_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Any):
    calls: dict[str, Any] = {}

    async def fake_run(model: Any, **kw: Any) -> str:
        calls['run'] = (model, kw)
        return 'id'

    monkeypatch.setattr(cli, 'run_shell', fake_run)
    monkeypatch.setattr(cli, 'configure_telemetry', lambda **kw: calls.setdefault('tel', kw) and False)
    monkeypatch.setenv('HELLO_PAX_MODEL', 'test')
    log = str(tmp_path / 'l.jsonl')
    argv = [
        '--no-logfire',
        '--log-file',
        log,
        '--logfire-content',
        '--thinking',
        'high',
        '--request-limit',
        '5',
        '--ephemeral',
        '--instructions',
        'hi',
    ]
    assert cli.main(argv) == 0
    model, kw = calls['run']
    assert model == 'test' and kw['thinking'] == 'high' and kw['request_limit'] == 5 and kw['ephemeral']
    assert kw['logfire_remote'] is False and kw['instructions'] == 'hi'
    assert calls['tel']['remote'] is False and calls['tel']['content'] is True


def test_interrupt(monkeypatch: pytest.MonkeyPatch):
    async def boom(*a: Any, **k: Any) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, 'run_shell', boom)
    assert cli.main(['-m', 'test', '--no-logfire']) == 130


def test_no_token_warns_and_reports_remote_off(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    seen: dict[str, Any] = {}

    async def fake_run(model: Any, **kw: Any) -> str:
        seen.update(kw)
        return 'id'

    monkeypatch.setattr(cli, 'run_shell', fake_run)
    monkeypatch.setattr(cli, 'configure_telemetry', lambda **kw: False)  # requested, but no token
    assert cli.main(['-m', 'test']) == 0
    assert seen['logfire_remote'] is False
    assert 'nothing is sent' in capsys.readouterr().err
