"""This module contains the `BaseAnalyzer` abstract base class shared by all analyzers.

Every analyzer accepts an optional calculator and lazily defaults it to a shared MLIP if
none is provided. `BaseAnalyzer` owns that bookkeeping so individual analyzers only need
to define what makes them distinct: their transformation(s) and their `calculate()` logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from materialsframework.utils import default_calculator, lazy_property, to_structure

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class BaseAnalyzer(ABC):
    """Shared base class for property analyzers.

    Owns lazy calculator management (`self._calculator` becomes `self.calculator`). Subclasses
    that need non-default calculator kwargs (e.g. a specific `fmax`) override the
    `calculator` property directly instead of overriding `__init__`.
    """

    def __init__(self, calculator: BaseCalculator | None = None) -> None:
        """Initializes the `BaseAnalyzer` object.

        Args:
            calculator (BaseCalculator | None, optional): The calculator used for the analysis.
                Defaults to a lazily constructed default calculator.
        """
        self._calculator = calculator

    @lazy_property("_calculator")
    def calculator(self) -> BaseCalculator:
        """Returns the calculator instance used for the analysis.

        If the calculator instance is not already initialized, this method returns the default
        calculator.

        Returns:
            BaseCalculator: The calculator object used for the analysis.
        """
        return default_calculator()

    def _ensure_relaxed(self, structure: Structure | Atoms, is_relaxed: bool) -> Structure:
        """Returns `structure` as a relaxed pymatgen `Structure`.

        Converts the input to a pymatgen `Structure` and, unless `is_relaxed` is `True`,
        relaxes it with `self.calculator` first.

        Args:
            structure (Structure | Atoms): The input structure.
            is_relaxed (bool): Whether the input structure is already relaxed.

        Returns:
            Structure: The (possibly relaxed) structure.
        """
        structure = to_structure(structure)

        if not is_relaxed:
            structure = self.calculator.relax(structure)["final_structure"]

        return structure

    @abstractmethod
    def calculate(self, *args, **kwargs) -> dict:
        """Runs the analysis and returns its results.

        Returns:
            dict: The analysis results.
        """
