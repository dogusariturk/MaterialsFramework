"""Generates deformed structures for Equation of State (EOS) calculations.

Applies uniform volumetric strains to an undeformed structure to produce the series of
structures used to fit an energy-volume equation of state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymatgen.transformations.standard_transformations import DeformStructureTransformation

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class EOSTransformation:
    """Generates deformed structures for EOS (Equation of State) calculations."""

    def __init__(
        self,
        start: float = -0.1,
        stop: float = 0.1,
        num: int = 11,
    ) -> None:
        """Initializes the `EOSTransformation` object.

        Args:
            start (float, optional): The starting strain value to apply to the structure. Defaults to -0.1.
            stop (float, optional): The stopping strain value to apply to the structure. Defaults to 0.1.
            num (int, optional): The number of strain values to generate between the start and stop. Defaults to 11.

        Note:
            The `start` and `stop` parameters define the range of strains to apply, while `num` determines how many
            evenly spaced strain values will be generated within that range.

        Raises:
            ValueError: If `num` is less than 3, since fitting an equation of state requires at least 3
                energy-volume points.
        """
        if num < 3:
            raise ValueError(f"num must be at least 3 to fit an equation of state, got {num}.")

        self._strains = np.linspace(start, stop, num)

    def apply_transformation(self, structure: Structure) -> list[Structure]:
        """Generate deformed structures for EOS calculations.

        Scales the lattice vectors of the input structure according to each strain value.

        Args:
            structure (Structure): The initial, undeformed structure to be used for EOS calculations.

        Returns:
            list[Structure]: The deformed structures generated for EOS calculations.
        """
        structures: list[Structure] = []
        for strain in self._strains:
            dst = DeformStructureTransformation((np.identity(3) * (1.0 + strain)).tolist())
            structures.append(dst.apply_transformation(structure))
        return structures
