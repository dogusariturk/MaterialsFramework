"""This module contains utility functions shared across the analysis, calculators, tools, and transformations packages."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


def default_calculator(**kwargs) -> BaseCalculator:
    """Returns a new `M3GNetCalculator` instance, the shared default calculator across the framework.

    Args:
        **kwargs: Keyword arguments forwarded to `M3GNetCalculator`.

    Returns:
        BaseCalculator: A new `M3GNetCalculator` instance.
    """
    from materialsframework.calculators.m3gnet import M3GNetCalculator

    return M3GNetCalculator(**kwargs)


def lazy_property(attr: str):
    """Wrap a method as a property whose return value is cached on ``self.<attr>``.

    The wrapped class must initialize ``attr`` to ``None`` (typically in ``__init__``);
    the wrapped method's body then runs only on first access, with its return value cached
    on that attribute and reused on subsequent accesses.

    Args:
        attr (str): Name of the instance attribute used as the cache slot.

    Returns:
        Callable: A decorator that turns the wrapped method into a caching property.
    """

    def decorator(func):
        @property
        @wraps(func)
        def wrapper(self):
            if getattr(self, attr) is None:
                setattr(self, attr, func(self))
            return getattr(self, attr)

        return wrapper

    return decorator


def to_structure(structure: Atoms | Structure) -> Structure:
    """Converts an `ase.Atoms` input to a pymatgen `Structure`, passing a `Structure` through unchanged.

    Args:
        structure (Atoms | Structure): The input structure.

    Returns:
        Structure: The structure as a pymatgen `Structure`.
    """
    if isinstance(structure, Atoms):
        return AseAtomsAdaptor().get_structure(structure)
    return structure


def to_atoms(structure: Atoms | Structure | Molecule) -> Atoms:
    """Converts a pymatgen `Structure`/`Molecule` input to `ase.Atoms`, copying an `Atoms` input unchanged.

    Args:
        structure (Atoms | Structure | Molecule): The input structure.

    Returns:
        Atoms: The structure as an `ase.Atoms` object.
    """
    if isinstance(structure, Atoms):
        return structure.copy()
    return AseAtomsAdaptor().get_atoms(structure)
