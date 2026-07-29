"""This module provides a class to calculate the formation energy of materials.

The `FormationEnergyAnalyzer` class computes the formation energy of a material based on its
composition and structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from materialsframework.analysis.base import BaseAnalyzer
from materialsframework.analysis.utils import require_properties
from materialsframework.transformations.formation_energy import FormationEnergyTransformation
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
        self._pure_reference_cache: dict[str, dict[str, float | bool | Structure]] = {}

    @require_properties("energy")
    def calculate(
        self, structure: Atoms | Structure, is_relaxed: bool = False
    ) -> dict[str, float | dict[str, dict[str, float | bool | Structure]]]:
        """Calculates the formation energy of the given structure.

        For elemental references, each element's known experimental ground-state structure
        (see `FormationEnergyTransformation`) is relaxed with the same calculator. For the
        few elements whose ground state can't be constructed directly, several candidate
        crystal structures are relaxed instead and the lowest energy per atom is used. Each
        element's reference energy is cached on this analyzer instance, so calling `calculate()`
        again (even on a different structure) reuses it instead of relaxing it again; construct
        a new analyzer to force fresh relaxations.

        Args:
            structure (Atoms | Structure): The structure for which the formation energy is calculated.
            is_relaxed (bool, optional): If ``True``, the structure is assumed to be already relaxed
                and only a single-point energy calculation is performed. Defaults to ``False``.

        Returns:
            dict[str, float]: Dictionary with keys:
                - ``formation_energy``: Formation energy per atom (eV/atom).
                - ``elemental_references``: Per-element dict of ``{"structure": Structure, "energy_per_atom": float,
                    "is_guessed": bool}``, where ``is_guessed`` is ``True`` if the element has no known experimental
                    ground state and a guessed high-symmetry candidate was used instead.
        """
        structure = to_structure(structure)

        if len(structure.composition.get_el_amt_dict()) < 2:
            raise ValueError("The structure must contain at least two different elements to calculate formation energy.")

        if is_relaxed:
            compound_energy = self.calculator.calculate(structure)["energy"]
        else:
            result = self.calculator.relax(structure)
            structure, compound_energy = result["final_structure"], result["energy"]

        pure_structures = self.formation_energy_transformation.apply_transformation(structure)

        elemental_references: dict[str, dict[str, float | bool | Structure]] = {}
        pure_energies = 0.0
        for element, candidates, num in pure_structures:
            if (reference := self._pure_reference_cache.get(element)) is None:
                relaxed = [self.calculator.relax(candidate) for candidate in candidates]
                energies_per_atom = [result["energy"] / result["final_structure"].num_sites for result in relaxed]
                best_index = energies_per_atom.index(min(energies_per_atom))
                reference = self._pure_reference_cache[element] = {
                    "structure": relaxed[best_index]["final_structure"],
                    "energy_per_atom": energies_per_atom[best_index],
                    "is_guessed": len(candidates) > 1,
                }
            elemental_references[element] = reference
            pure_energies += num * reference["energy_per_atom"]

        return {
            "formation_energy": (compound_energy - pure_energies) / structure.num_sites,
            "elemental_references": elemental_references,
        }

    @lazy_property("_formation_energy_transformation")
    def formation_energy_transformation(self) -> FormationEnergyTransformation:
        """Returns the transformation object used to apply distortions.

        Returns:
            FormationEnergyTransformation: The transformation object used to generate structures.
        """
        return FormationEnergyTransformation()
