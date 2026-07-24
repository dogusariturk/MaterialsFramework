"""This module provides a class to calculate the formation energy of materials.

The `FormationEnergyAnalyzer` class computes the formation energy of a material based on its
composition and structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from materialsframework.analysis.base import BaseAnalyzer
from materialsframework.analysis.utils import require_properties
from materialsframework.transformations.formation_energy import (
    FormationEnergyTransformation,
)
from materialsframework.utils import lazy_property, to_structure

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class FormationEnergyAnalyzer(BaseAnalyzer):
    """A class used to calculate the formation energy of materials.

    The `FormationEnergyAnalyzer` class computes the formation energy of a material from its
    composition and structure.
    """

    def __init__(
        self,
        calculator: BaseCalculator | None = None,
        formation_energy_transformation: FormationEnergyTransformation | None = None,
    ) -> None:
        """Initializes the `FormationEnergyAnalyzer` object.

        Args:
            calculator (BaseCalculator | None, optional): The calculator object used for energy calculations.
            formation_energy_transformation (FormationEnergyTransformation, optional): The transformation
                object used to generate structures required for the calculation of formation energies.
        """
        super().__init__(calculator)
        self._formation_energy_transformation = formation_energy_transformation

    @require_properties("energy")
    def calculate(self, structure: Atoms | Structure, is_relaxed: bool = False) -> dict[str, float]:
        """Calculates the formation energy of the given structure.

        For elemental references, three candidate crystal structures (FCC, BCC, HCP) are
        relaxed with the same calculator for each element and the lowest energy per atom
        is used.

        Args:
            structure (Atoms | Structure): The structure for which the formation energy is calculated.
            is_relaxed (bool, optional): If ``True``, the structure is assumed to be already relaxed
                and only a single-point energy calculation is performed. Defaults to ``False``.

        Returns:
            dict[str, float]: Dictionary with keys:
                - ``formation_energy``: Formation energy per atom (eV/atom).
        """
        structure = to_structure(structure)

        if len(structure.elements) < 2:
            raise ValueError("The structure must contain at least two different elements to calculate formation energy.")

        if is_relaxed:
            compound_energy = self.calculator.calculate(structure)["energy"]
        else:
            result = self.calculator.relax(structure)
            compound_energy = result["energy"]
            structure = result["final_structure"]

        pure_structures = self.formation_energy_transformation.apply_transformation(structure)

        pure_energies = sum(
            num * min(self.calculator.relax(candidate)["energy"] / candidate.num_sites for candidate in candidates)
            for candidates, num in pure_structures
        )

        return {
            "formation_energy": (compound_energy - pure_energies) / structure.num_sites,
        }

    @lazy_property("_formation_energy_transformation")
    def formation_energy_transformation(self) -> FormationEnergyTransformation:
        """Returns the transformation object used to apply distortions.

        Returns:
            FormationEnergyTransformation: The transformation object used to generate structures.
        """
        return FormationEnergyTransformation()
