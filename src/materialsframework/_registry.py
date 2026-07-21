"""Shared entry-point-registry helpers used by the calculators, analysis, transformations, and tools packages.

Each of those packages exposes the same `list_x()`/`get_x()` registry pair backed by an
`importlib.metadata` entry-point group, plus a package `__getattr__` for lazy, attribute-style
access to its classes. This module factors out that shared template.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


def make_registry(group: str, label: str) -> tuple[Callable[[], list[str]], Callable[..., Any]]:
    """Build a `list_x`/`get_x` pair backed by an `importlib.metadata` entry-point group.

    Args:
        group (str): Entry-point group name, e.g. "materialsframework.analyzers".
        label (str): Singular noun used in the "Unknown <label> ..." error message, e.g. "analyzer".

    Returns:
        tuple[Callable[[], list[str]], Callable[..., Any]]: A `(list_x, get_x)` pair, where `list_x()` returns the sorted
            registered names and `get_x(name, **kwargs)` instantiates the entry registered under `name`, forwarding `**kwargs` to
            its `__init__`.
    """

    def list_x() -> list[str]:
        return sorted(ep.name for ep in entry_points(group=group))

    def get_x(name: str, **kwargs) -> Any:
        eps = {ep.name: ep for ep in entry_points(group=group)}
        if name not in eps:
            raise ValueError(f"Unknown {label} {name!r}. Available: {', '.join(sorted(eps))}")
        return eps[name].load()(**kwargs)

    return list_x, get_x


def lazy_getattr(name: str, module_name: str, class_map: dict[str, tuple[str, str]]) -> type:
    """Resolve `name` to a class via `class_map`, importing its module on demand.

    Used to implement a package's `__getattr__` for attribute-style access to lazily-imported
    classes (e.g. `analysis.EOSAnalyzer`) without eagerly importing every submodule.

    Args:
        name (str): The attribute name being looked up.
        module_name (str): The importing package's `__name__`, used in the `AttributeError` message.
        class_map (dict[str, tuple[str, str]]): Maps a class name to its `(module_path, class_name)`.

    Returns:
        type: The resolved class.

    Raises:
        AttributeError: If `name` is not a key in `class_map`.
    """
    if name in class_map:
        module_path, class_name = class_map[name]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
