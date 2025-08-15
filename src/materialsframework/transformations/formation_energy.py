"""
This module provides a class to generate structures for formation energy calculations.

The `FormationEnergyTransformation` class facilitates the generation of structures
required for the calculation of formation energies.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ase import Atoms
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class FormationEnergyTransformation:
    """
    A class used to generate structures for formation energy calculations.

    The `FormationEnergyTransformation` class provides methods to generate structures
    for the calculation of formation energies.

    Attributes:
        pure_structures (list): A list to store the pure structures generated for formation energy calculations
    """
    def __init__(self):
        """
        Initializes the `FormationEnergyTransformation` object.
        """
        self.ase_adaptor = AseAtomsAdaptor()
        self.pure_structures: list[tuple[Atoms, int]] = []

    def apply_transformation(self, structure: Atoms | Structure) -> None:
        """
        Apply the transformation to calculate the formation energy of the given structure.

        Args:
            structure (Structure): The structure to apply the transformation.
        """
        if isinstance(structure, Atoms):
            structure = self.ase_adaptor.get_structure(structure)

        for element, num in structure.composition.get_el_amt_dict().items():
            atoms = bulk(str(element))  # TODO: Refactor this for elements without a basis in the reference_states dict
            self.pure_structures.append((atoms, int(num)))
