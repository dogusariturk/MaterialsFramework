"""This module contains utility functions shared across analyzer implementations."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


def require_properties(*properties: str) -> Callable:
    """Decorator raising before the wrapped method runs if `self.calculator` lacks any of `properties`.

    Args:
        *properties (str): Property names that must all be present in `self.calculator.AVAILABLE_PROPERTIES`.

    Returns:
        Callable: A decorator that wraps a bound method, checking `self.calculator.AVAILABLE_PROPERTIES`
            before running the method's body.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not set(properties).issubset(self.calculator.AVAILABLE_PROPERTIES):
                names = " and ".join(f"'{prop}'" for prop in properties)
                noun = "property" if len(properties) == 1 else "properties"
                raise ValueError(f"The calculator object must have the {names} {noun} implemented.")
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
