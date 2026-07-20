"""Transformation registry backed by the materialsframework.transformations entry-point group."""

from __future__ import annotations

from typing import TYPE_CHECKING

from materialsframework._registry import make_registry

if TYPE_CHECKING:
    from typing import Any

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

_list_transformations, _get_transformation = make_registry("materialsframework.transformations", "transformation")


def list_transformations() -> list[str]:
    """Return sorted names of all registered transformations.

    Returns:
        Sorted list of registered transformation names.
    """
    return _list_transformations()


def get_transformation(name: str, **kwargs) -> Any:
    """Instantiate a transformation by its registered name.

    Args:
        name: Registered transformation name (e.g. "eos", "bain", "phonopy").
        **kwargs: Forwarded to the transformation's ``__init__``.

    Returns:
        An initialized transformation instance.

    Raises:
        ValueError: If no transformation is registered under the given name.
    """
    return _get_transformation(name, **kwargs)
