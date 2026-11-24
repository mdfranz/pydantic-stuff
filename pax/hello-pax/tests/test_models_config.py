import pytest
from pydantic_ai.models.openai import OpenAIChatModel

from hello_pax.models_config import (
    GATEWAY_BASE_URL_ENV,
    ModelConfigError,
    model_display_name,
    normalize_gateway_base_url,
    resolve_model,
)


@pytest.mark.parametrize(
    'm', ['google:gemini-3.8-flash', 'gateway/anthropic:claude-sonnet-5-5', 'openai:gpt-4o', 'test']
)
def test_passthrough(m: str):
    assert resolve_model(f'  {m} ') == m


@pytest.mark.parametrize(
    ('model', 'route'), [('', None), ('  ', None), ('bare', None), ('openai:x', 'r'), ('bare', 'a/b'), ('bare', ' ')]
)
def test_errors(model: str, route: str | None):
    with pytest.raises(ModelConfigError) as ei:
        resolve_model(model, gateway_route=route)
    assert ei.value.rule_code == 'model-config'


def test_missing_gateway_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('PYDANTIC_AI_GATEWAY_API_KEY', raising=False)
    with pytest.raises(ModelConfigError):
        resolve_model('gpt-4o', gateway_route='r')


def test_routed_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PYDANTIC_AI_GATEWAY_API_KEY', 'k')
    monkeypatch.setenv(GATEWAY_BASE_URL_ENV, 'https://gw.example/proxy/doubleword/')
    model = resolve_model('gpt-4o', gateway_route='doubleword')
    assert isinstance(model, OpenAIChatModel)
    assert model_display_name(model) == 'gpt-4o'
    assert model_display_name('test') == 'test'


@pytest.mark.parametrize('base', ['https://gw/proxy/doubleword', 'https://gw/proxy/doubleword/', 'https://gw/proxy'])
def test_normalize(base: str):
    env = {GATEWAY_BASE_URL_ENV: base}
    normalize_gateway_base_url('doubleword', env)
    assert env[GATEWAY_BASE_URL_ENV] == 'https://gw/proxy'
