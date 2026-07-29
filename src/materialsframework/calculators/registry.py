"""Calculator registry backed by the materialsframework.calculators entry-point group."""

from __future__ import annotations

from typing import TYPE_CHECKING

from materialsframework._registry import make_registry

if TYPE_CHECKING:
    from typing import Any

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

_list_calculators, _get_calculator = make_registry("materialsframework.calculators", "calculator")


def list_calculators() -> list[str]:
    """Return sorted names of all registered calculators.

    Returns:
        Sorted list of registered calculator names.
    """
    return _list_calculators()


def get_calculator(name: str, **kwargs: Any) -> BaseCalculator:
    """Instantiate a calculator by its registered name.

    Args:
        name: Registered calculator name (e.g. "mace", "chgnet").
        **kwargs: Forwarded to the calculator's ``__init__``.

    Returns:
        An initialized calculator instance.

    Raises:
        ValueError: If no calculator is registered under the given name.
    """
    return _get_calculator(name, **kwargs)
