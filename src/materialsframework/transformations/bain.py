"""Generates displaced structures along the Bain path.

Produces structures deformed along the Bain path through a continuous change in the
c/a ratio, useful for studying martensitic transformations between cubic and tetragonal
phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class BainDisplacementTransformation:
    """Generates a series of deformed structures along the Bain path.

    Varies the c/a ratio in small steps to produce structures along the Bain transformation
    pathway between cubic and tetragonal phases.
    """

    def __init__(
        self,
        start: float = 0.89,
        stop: float = 1.5,
        step: float = 0.01,
    ) -> None:
        """Initializes the `BainDisplacementTransformation` object.

        Args:
            start (float, optional): The starting displacement value for the c/a ratio. Defaults to 0.89.
            stop (float, optional): The stopping displacement value for the c/a ratio. Defaults to 1.5.
            step (float, optional): The step size for incrementing the c/a ratio. Defaults to 0.01.
        """
        self.c_a_ratios: np.ndarray = np.arange(start=start, stop=stop, step=step)

    def apply_transformation(
        self,
        structure: Structure,
    ) -> dict[float, Structure]:
        """Generate displaced structures along the Bain path.

        Args:
            structure (Structure): The input structure to be displaced along the Bain path.

        Returns:
            dict[float, Structure]: The displaced structures, keyed by the corresponding c/a ratio.
        """
        displaced_structures: dict[float, Structure] = {}
        for c_a in self.c_a_ratios:
            delta = np.cbrt(1 / c_a) - 1
            displaced_structures[c_a] = self._get_displaced_structures(delta, structure)
        return displaced_structures

    @staticmethod
    def _get_displaced_structures(delta: float, structure: Structure) -> Structure:
        """Apply a Bain-path deformation to the structure for a given delta.

        Args:
            delta (float): The displacement value for the Bain transformation.
            structure (Structure): The input structure to be deformed.

        Returns:
            Structure: The displaced structure with the Bain transformation applied.
        """
        transformation_matrix = [
            [1 + delta, 0, 0],
            [0, 1 + delta, 0],
            [0, 0, 1 / (1 + delta) ** 2],
        ]

        deformation = DeformStructureTransformation(deformation=transformation_matrix)
        return deformation.apply_transformation(structure)
