"""Analyzer registry backed by the materialsframework.analyzers entry-point group."""

from __future__ import annotations

from typing import TYPE_CHECKING

from materialsframework._registry import make_registry

if TYPE_CHECKING:
    from typing import Any

    from materialsframework.analysis.base import BaseAnalyzer

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

_list_analyzers, _get_analyzer = make_registry("materialsframework.analyzers", "analyzer")


def list_analyzers() -> list[str]:
    """Return sorted names of all registered analyzers.

    Returns:
        Sorted list of registered analyzer names.
    """
    return _list_analyzers()


def get_analyzer(name: str, **kwargs: Any) -> BaseAnalyzer:
    """Instantiate an analyzer by its registered name.

    Args:
        name: Registered analyzer name (e.g. "eos", "bain", "phonopy").
        **kwargs: Forwarded to the analyzer's ``__init__``.

    Returns:
        An initialized analyzer instance.

    Raises:
        ValueError: If no analyzer is registered under the given name.
    """
    return _get_analyzer(name, **kwargs)
