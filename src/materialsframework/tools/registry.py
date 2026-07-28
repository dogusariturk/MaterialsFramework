"""Tool registry backed by the materialsframework.tools entry-point group."""

from __future__ import annotations

from typing import TYPE_CHECKING

from materialsframework._registry import make_registry

if TYPE_CHECKING:
    from typing import Any

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

_list_tools, _get_tool = make_registry("materialsframework.tools", "tool")


def list_tools() -> list[str]:
    """Return sorted names of all registered tools.

    Returns:
        Sorted list of registered tool names.
    """
    return _list_tools()


def get_tool(name: str, **kwargs: Any) -> Any:
    """Instantiate a tool by its registered name.

    Args:
        name: Registered tool name (e.g. "cluster_expansion", "stability_map").
        **kwargs: Forwarded to the tool's ``__init__``.

    Returns:
        An initialized tool instance.

    Raises:
        ValueError: If no tool is registered under the given name.
    """
    return _get_tool(name, **kwargs)
