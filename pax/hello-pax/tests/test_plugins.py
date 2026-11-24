import importlib.util

from hello_pax.agent_adapter import STOCK_PLUGINS
from hello_pax.agent_adapter.search_plugin import SEARCH_PLUGIN_ID
from hello_pax.agent_adapter.status_plugin import STATUS_PLUGIN_ID

DENIED = {'coder', 'sandbox', 'workspaces', 'repo_context', 'hooks', 'mcp', 'observability', 'notifications'}


def test_stock_plugin_declarations():
    assert {p.id: p.factory for p in STOCK_PLUGINS} == {
        'refinement': 'pydantic_clai2.builtin_plugins.refinement',
        'compaction': 'pydantic_clai2.builtin_plugins.compaction',
        'ask_user': 'pydantic_clai2.builtin_plugins.ask_user_menu',
    }


def test_factories_resolve():
    for plugin in STOCK_PLUGINS:
        assert importlib.util.find_spec(plugin.factory.split(':')[0]) is not None, plugin.id


def test_no_filesystem_or_shell_plugins():
    ids = {p.id for p in STOCK_PLUGINS}
    assert not ids & DENIED
    assert len(ids | {STATUS_PLUGIN_ID, SEARCH_PLUGIN_ID}) == len(ids) + 2  # no id clashes
