"""
This module provides a class to perform a Bain transformation on a given structure.
"""
from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.calculators import Calculator

from materialsframework.calculators import M3GNetCalculator
from materialsframework.transformations import BainDisplacementTransformation

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class BainPathAnalyzer:
    """
    A class used to represent a BainPathAnalyzer.

    This class provides methods to perform a Bain transformation on a given structure.
    """

    def __init__(
            self,
            calculator: Optional[Calculator] = None,
            bain_transformation: Optional[BainDisplacementTransformation] = None
    ) -> None:
        """
        Initializes the BainPathAnalyzer.

        Parameters:
            calculator (Optional[Calculator]): The calculator object to use for calculating potential energies.
            bain_transformation (Optional[BainDisplacementTransformation]): The bain displacement transformation object.
        """
        self._calculator = calculator
        self._bain_transformation = bain_transformation

    def calculate(self, undeformed_structure: Structure) -> dict:
        """
        Calculates the potential energies along the Bain Path for the given undeformed structure.

        Parameters:
            undeformed_structure (Structure): The undeformed relaxed structure.

        Returns:
            dict: A dictionary containing the c_a ratios and calculated potential energies along the Bain Path.
        """
        self.bain_transformation.apply_transformation(structure=undeformed_structure)

        c_a_list, energy_list = zip(
                *[(c_a, self.calculator.calculate(structure=deformed_structure)["potential_energy"])
                  for c_a, deformed_structure in self.bain_transformation.displaced_structures.items()])

        return {
                "c_a_list": c_a_list,
                "energy_list": energy_list
        }

    @property
    def calculator(self) -> Calculator:
        """
        Gets the calculator used for calculating potential energies.
        If not set, initializes a new M3GNetCalculator.

        Returns:
            Calculator: The calculator object.
        """
        if self._calculator is None:
            self._calculator = M3GNetCalculator()
        return self._calculator

    @property
    def bain_transformation(self) -> BainDisplacementTransformation:
        """
        Gets the bain displacement transformation object.

        Returns:
            BainDisplacementTransformation: The bain displcement transformation object.
        """
        if self._bain_transformation is None:
            self._bain_transformation = BainDisplacementTransformation()
        return self._bain_transformation
