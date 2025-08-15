"""
This module contains a class to calculate the formation energy of materials.

The `FormationEnergyAnalyzer` class computes the formation energy of a material based on its
composition and structure. The class can be used to analyze the stability of materials and
determine their suitability for various applications.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from materialsframework.transformations.formation_energy import FormationEnergyTransformation

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class FormationEnergyAnalyzer:
    """
    A class used to calculate the formation energy of materials.
    The `FormationEnergyAnalyzer` class provides methods to compute the formation energy of a
    material based on its composition and structure. The class can be used to analyze the stability
    of materials and determine their suitability for various applications.
    """

    def __init__(
            self,
            calculator: BaseCalculator | None = None,
            formation_energy_transformation: FormationEnergyTransformation | None = None
    ) -> None:
        """
        Initializes the `FormationEnergyAnalyzer` object.

        Args:
             calculator (BaseCalculator | None, optional): The calculator object used for energy calculations.
                                                              Defaults to `M3GNetCalculator`.
            formation_energy_transformation (FormationEnergyTransformation, optional): The transformation
                object used to generate structures required for the calculation of formation energies.
        """
        self.ase_adaptor = AseAtomsAdaptor()
        self._calculator = calculator
        self._formation_energy_transformation = formation_energy_transformation

    def calculate(
            self,
            structure: Atoms | Structure,
            is_relaxed: bool = False
    ) -> dict[str, float]:
        """
        Calculates the formation energy of the given structure.

        Args:
            structure (Atoms | Structure): The structure for which the formation energy is calculated.
            is_relaxed (bool, optional): If True, the structure is relaxed before calculation. Defaults to False.

        Returns:
            dict[str, float]: A dictionary containing the formation energy of the structure.
        """
        if "energy" not in self.calculator.AVAILABLE_PROPERTIES:
            raise ValueError("The calculator object must have the 'energy' property implemented.")

        if isinstance(structure, Atoms):
            structure = self.ase_adaptor.get_structure(structure)

        if len(structure.elements) < 2:
            raise ValueError("The structure must contain at least two different elements to calculate formation energy.")

        if not is_relaxed:
            structure = self.calculator.relax(structure)["final_structure"]

        initial_structure = structure.copy()
        initial_energy = self.calculator.calculate(initial_structure)["energy"]

        self.formation_energy_transformation.apply_transformation(structure)
        pure_energies = sum(num * (self.calculator.relax(pure_structure)["energy"] / len(pure_structure))
                            for pure_structure, num in self.formation_energy_transformation.pure_structures)

        return {
                "formation_energy": (initial_energy - pure_energies) / len(initial_structure),
        }

    @property
    def calculator(self) -> BaseCalculator:
        """
        Returns the calculator instance used for energy calculations.

        If the calculator instance is not already initialized, this method creates a new `M3GNetCalculator` instance.

        Returns:
            BaseCalculator: The calculator object used for energy calculations.
        """
        if self._calculator is None:
            from materialsframework.calculators.m3gnet import M3GNetCalculator
            self._calculator = M3GNetCalculator(fmax=0.01)
        return self._calculator

    @property
    def formation_energy_transformation(self) -> FormationEnergyTransformation:
        """
        Returns the transformation object used to apply distortions.

        If the transformation object is not already initialized, this method creates a new `FormationEnergyTransformation` instance.

        Returns:
            FormationEnergyTransformation: The transformation object used to generate structures.
        """
        if self._formation_energy_transformation is None:
            self._formation_energy_transformation = FormationEnergyTransformation()
        return self._formation_energy_transformation
