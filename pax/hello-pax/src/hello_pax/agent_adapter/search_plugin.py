from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any

from pydantic_clai2.commands import Command  # pyright: ignore[reportMissingTypeStubs]
from pydantic_clai2.plugins import NoSettings, Plugin  # pyright: ignore[reportMissingTypeStubs]

from ..errors import HelloPaxError
from .search import DEFAULT_MODELS, PROVIDERS, WebSearcher, is_provider, missing_key, search_model

SEARCH_PLUGIN_ID = 'hello_pax_search'
SEARCH_PLUGIN_FACTORY = 'hello_pax.agent_adapter.search_plugin:SearchPlugin'
USAGE = 'usage: /search <google|anthropic|openai> <query>   (or: /search --help)'


class SearchControl:
    """Operator-only ``/search``: runs native web search on a chosen provider and prints the result.

    Never sent to the shell agent: the answer is shown to the operator and not added to the conversation.
    """

    def __init__(self, searcher: WebSearcher | None = None) -> None:
        self.searcher = searcher or WebSearcher()

    def completions(self, args: list[str]) -> tuple[str, ...]:
        return PROVIDERS if len(args) <= 1 else ()

    def help(self) -> str:
        rows = [
            f'  {p:<10} {search_model(p)}  [{"ready" if missing_key(p) is None else "needs " + str(missing_key(p))}]'
            for p in PROVIDERS
        ]
        return '\n'.join([USAGE, 'providers:', *rows, f'{self.searcher.calls_used}/{self.searcher.max_calls} used'])

    async def handle(self, args: list[str]) -> str:
        text = args[0].strip() if args else ''
        if not text or text in {'--help', '-h'}:
            return self.help()
        provider, _, query = text.partition(' ')
        provider = provider.lower()
        if not is_provider(provider):
            return f'unknown provider {provider!r}\n{USAGE}'
        try:
            result = await self.searcher.search(provider, query)  # pyright: ignore[reportArgumentType]
        except HelloPaxError as exc:
            return f'search: {exc}'
        return result.render()


current_search: ContextVar[SearchControl | None] = ContextVar('hello_pax_search_control', default=None)


class SearchPlugin(Plugin[NoSettings]):
    def get_capabilities(self) -> Sequence[Any]:
        return ()

    def get_commands(self) -> Sequence[Command]:
        control = current_search.get()
        if control is None:
            return ()
        return (
            Command(
                name='search',
                description='Web search via google, anthropic, or openai: /search <provider> <query>',
                handler=control.handle,
                complete=control.completions,
                raw=True,
                live=True,
            ),
        )


__all__ = [
    'DEFAULT_MODELS',
    'SEARCH_PLUGIN_FACTORY',
    'SEARCH_PLUGIN_ID',
    'SearchControl',
    'SearchPlugin',
    'current_search',
]
