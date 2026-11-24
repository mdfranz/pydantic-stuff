import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from hello_pax.agent_adapter.search import (
    SEARCH_QUERY_MAX_CHARS,
    SearchError,
    WebSearcher,
    build_search_agent,
    extract_sources,
    missing_key,
    search_model,
    validate_query,
)
from hello_pax.agent_adapter.search_plugin import SearchControl, SearchPlugin, current_search

KEYS = {'GEMINI_API_KEY': 'k', 'ANTHROPIC_API_KEY': 'k', 'OPENAI_API_KEY': 'k'}


def stubbed(text: str = 'the answer') -> WebSearcher:
    searcher = WebSearcher()
    for provider in ('google', 'anthropic', 'openai'):
        searcher._agents[provider] = Agent(TestModel(custom_output_text=text))  # pyright: ignore[reportPrivateUsage]
    return searcher


def test_validate_query():
    assert validate_query('  a   b\n c ') == 'a b c'
    for bad in ('', '   ', 'x' * (SEARCH_QUERY_MAX_CHARS + 1), 'see https://u:p@host/x', 'see https://h/x?token=1'):
        with pytest.raises(SearchError):
            validate_query(bad)
    assert validate_query('see https://example.com/docs') == 'see https://example.com/docs'


def test_models_and_keys():
    assert search_model('google', {}) == 'google:gemini-3.8-flash'
    assert search_model('openai', {}) == 'openai-responses:gpt-5.5'
    assert search_model('openai', {'HELLO_PAX_SEARCH_OPENAI_MODEL': ' openai:o9 '}) == 'openai:o9'
    assert missing_key('google', {'GOOGLE_API_KEY': 'x'}) is None
    assert missing_key('openai', {}) == 'OPENAI_API_KEY'
    assert missing_key('google', {}) == 'GEMINI_API_KEY or GOOGLE_API_KEY'


def test_build_agent_without_keys():
    for provider, model in (('google', 'google:g'), ('anthropic', 'anthropic:c'), ('openai', 'openai:o')):
        assert build_search_agent(model, provider=provider).name == 'hello_pax_search_agent'  # pyright: ignore[reportArgumentType]


def test_extract_sources_empty():
    assert extract_sources([]) == []


async def test_search_ok_and_cap():
    searcher = stubbed()
    searcher.max_calls = 1
    result = await searcher.search('google', 'what is pydantic', env=KEYS)
    assert result.answer == 'the answer' and 'google' in result.render()
    with pytest.raises(SearchError, match='cap'):
        await searcher.search('google', 'again', env=KEYS)


async def test_search_missing_key_does_not_count():
    searcher = stubbed()
    with pytest.raises(SearchError, match='OPENAI_API_KEY'):
        await searcher.search('openai', 'q', env={})
    assert searcher.calls_used == 0


async def test_control():
    c = SearchControl(stubbed())
    assert 'usage' in await c.handle([])
    assert 'usage' in await c.handle(['--help'])
    assert 'unknown provider' in await c.handle(['bing q'])
    assert 'search: ' in await c.handle(['google '])
    assert c.completions([]) == ('google', 'anthropic', 'openai')
    assert c.completions(['google', 'x']) == ()


def test_plugin_binding():
    plugin = SearchPlugin.__new__(SearchPlugin)
    assert plugin.get_commands() == () and plugin.get_capabilities() == ()
    token = current_search.set(SearchControl())
    try:
        (command,) = plugin.get_commands()
        assert command.name == 'search' and command.raw
    finally:
        current_search.reset(token)
