"""Base classes for calculators, relaxers"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class Calculator(ABC):
    """Base class for all calculators"""

    @property
    @abstractmethod
    def potential(self):
        """Returns the potential associated with this calculator object."""

    @abstractmethod
    def calculate(self, structure: Structure):
        """Performs the calculation on the input structure and returns the results."""


class Relaxer(ABC):
    """Base class for all relaxers"""

    @property
    @abstractmethod
    def potential(self):
        """Returns the potential associated with this relaxer object."""

    @property
    @abstractmethod
    def relaxer(self):
        """Returns the relaxer associated with this instance."""

    @abstractmethod
    def relax(self, structure: Structure):
        """Relaxes the input structure and returns the relaxed structure and other information."""


class MDCalculator(ABC):
    """Base class for all molecular dynamics (MD) calculators"""

    @property
    @abstractmethod
    def potential(self):
        """Returns the potential associated with this calculator object."""

    @abstractmethod
    def run(self, structure: Structure, steps: int):
        """Runs a simulation on the input structure with the given number of steps and returns the results."""
