"""This module contains utility functions shared across the analysis, calculators, tools, and transformations packages."""

from __future__ import annotations

import importlib.util
from functools import wraps
from typing import TYPE_CHECKING

from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from collections.abc import Callable

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


def requires(*packages: str, extra: str | None = None, hint: str | None = None) -> Callable:
    """Decorator raising a clear `ImportError` before the wrapped callable runs if any of `packages` is not importable.

    Args:
        *packages (str): Top-level module names that must be importable (e.g. "pycalphad").
        extra (str | None, optional): Name of this package's optional-dependency extra that provides
            `packages`, used to build the install hint as `pip install materialsframework[<extra>]`.
            Ignored if `hint` is given. Defaults to None.
        hint (str | None, optional): Install instructions to use verbatim, for packages that aren't
            installable via a `materialsframework` extra (e.g. git-only dependencies). Takes precedence
            over `extra`. If neither is given, the hint falls back to `pip install <packages>`.
            Defaults to None.

    Returns:
        Callable: A decorator that checks importability of `packages` before running the wrapped
            function or method.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            missing = [pkg for pkg in packages if importlib.util.find_spec(pkg) is None]
            if missing:
                names = " and ".join(f"'{pkg}'" for pkg in missing)
                verb = "is" if len(missing) == 1 else "are"
                pronoun = "it" if len(missing) == 1 else "them"
                if hint:
                    install_hint = hint
                elif extra:
                    install_hint = f"pip install materialsframework[{extra}]"
                else:
                    install_hint = f"pip install {' '.join(missing)}"
                raise ImportError(f"{names} {verb} required. Install {pronoun} with: {install_hint}")
            return func(*args, **kwargs)

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
