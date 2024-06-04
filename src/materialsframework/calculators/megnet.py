"""
This module provides classes to perform calculations using the MEGNet potential.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import matgl

from materialsframework.calculators.typing import Calculator

if TYPE_CHECKING:
    from matgl.models import MEGNet
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class MEGNetCalculator(Calculator):
    """
    A class used to represent a MEGNet Calculator.

    This class provides methods to perform calculations using the MEGNet potential.
    """

    def __init__(self, model: str = "MEGNet-MP-2018.6.1-Eform"):
        """
        Initialize a MEGNetCalculator instance.

        Args:
            model (str): The name of the MEGNet model to use.

        Examples:
            >>> calculator = MEGNetCalculator()
            >>> calculator = MEGNetCalculator("MEGNet-MP-2018.6.1-Eform")

        Note:
            The remaining values for the arguments are set to the default values for the MEGNet potential.
        """
        self._model: str = model

        self._potential = None

    @property
    def potential(self) -> MEGNet:
        """
        Returns the MEGNet potential associated with this instance.

        If the potential has not been initialized yet, it will be loaded
        using the model attribute of this instance.

        Returns:
            MEGNet: The MEGNet potential associated with this instance.
        """
        if self._potential is None:
            self._potential = matgl.load_model(self._model)
        return self._potential

    def calculate(self, structure: Structure) -> dict[str, float]:
        """
        Calculate the formation energy of a structure using the MEGNet potential.

        Args:
            structure: The input structure.

        Returns:
            dict[str, float]: A dictionary containing the formation energy of the structure.

        Examples:
            >>> calculator = MEGNetCalculator()
            >>> struct = Structure.from_file("POSCAR")
            >>> calculation_results = calculator.calculate(structure=struct)
        """
        return {
                "formation_energy": self.potential.predict_structure(structure)
        }
