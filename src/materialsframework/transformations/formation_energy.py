"""
This module provides a class to generate structures for formation energy calculations.

The `FormationEnergyTransformation` class facilitates the generation of structures
required for the calculation of formation energies.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ase import Atoms
from ase.build import bulk

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class FormationEnergyTransformation:
    """
    A class used to generate structures for formation energy calculations.

    The `FormationEnergyTransformation` class provides methods to generate structures
    for the calculation of formation energies.
    """
    def __init__(self):
        """
        Initializes the `FormationEnergyTransformation` object.
        """
        self.pure_structures: dict[float, Atoms] = {}

    def apply_transformation(self, structure: Structure) -> None:
        """
        Apply the transformation to calculate the formation energy of the given structure.

        Args:
            structure (Structure): The structure to apply the transformation.
        """
        for element, comp in structure.composition.items():
            atoms = bulk(str(element), cubic=True)  # TODO: Fix this for elements without a basis in the reference_states dict
            self.pure_structures[comp] = atoms
