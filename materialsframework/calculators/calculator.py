"""Base class for all calculators"""

from abc import ABC, abstractmethod

from pymatgen.core import Structure


class Calculator(ABC):
    """Base class for all calculators"""

    @property
    @abstractmethod
    def potential(self):
        """Returns the potential associated with this calculator object."""

    @abstractmethod
    def calculate(self, structure: Structure):
        """Performs the calculation on the input structure and returns the results."""
