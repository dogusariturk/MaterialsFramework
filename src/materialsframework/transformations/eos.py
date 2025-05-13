"""
This module contains the `EOSTransformation` class that generates deformed structures for Equation of State (EOS) calculations.

The `EOSTransformation` class provides methods to systematically deform structures by varying the volume.
These deformed structures are used to calculate the equation of state, which describes the relationship
between volume, pressure, and energy of a material system.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class EOSTransformation:
    """
    A class used to generate deformed structures for EOS (Equation of State) calculations.

    The `EOSTransformation` class generates a series of deformed structures by applying
    uniform volumetric strains to an undeformed structure. These deformed structures can
    be used to calculate the equation of state.
    """

    def __init__(
            self,
            start: float = -0.1,
            stop: float = 0.1,
            num: int = 11,
    ) -> None:
        """
        Initializes the `EOSTransformation` object.

        Args:
            start (float, optional): The starting strain value to apply to the structure. Defaults to -0.1.
            stop (float, optional): The stopping strain value to apply to the structure. Defaults to 0.1.
            num (int, optional): The number of strain values to generate between the start and stop. Defaults to 11.

        Note:
            The strains are applied as uniform volumetric deformations to the structure, meaning
            the volume is scaled while preserving the shape of the unit cell.
        """
        self._strains = np.linspace(start, stop, num)

        self.structures: dict[float, Structure] = {}

    def apply_transformation(
            self,
            structure: Structure
    ) -> None:
        """
        Applies the transformation to generate deformed structures for EOS calculations.

        This method generates a series of deformed structures by scaling the volume of the input structure.
        The resulting deformed structures are stored in the `structures` attribute, keyed by the corresponding
        strain value.

        Args:
            structure (Structure): The initial, undeformed structure to be used for EOS calculations.

        Note:
            The volume of the structure is scaled by `(1 + strain)` for each strain value in the `_strains` array.
            This method does not modify the input structure; it creates copies that are scaled and stored.
        """
        initial_volume = structure.volume

        for strain in self._strains:
            structure = structure.copy()
            deformed_structure = structure.scale_lattice(initial_volume * (1 + strain))
            self.structures[strain] = deformed_structure
