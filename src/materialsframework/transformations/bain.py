"""Generates displaced structures along the Bain path.

Produces structures deformed along the Bain path through a continuous change in the
c/a ratio, useful for studying martensitic transformations between cubic and tetragonal
phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
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
        symprec: float = 1e-2,
    ) -> None:
        """Initializes the `BainDisplacementTransformation` object.

        Args:
            start (float, optional): The starting displacement value for the c/a ratio. Defaults to 0.89.
            stop (float, optional): The stopping displacement value for the c/a ratio. Defaults to 1.5.
            step (float, optional): The step size for incrementing the c/a ratio. Defaults to 0.01.
            symprec (float, optional): Symmetry precision used to standardize the input structure to a
                conventional cell before deforming it. Defaults to 1e-2.

        Raises:
            ValueError: If `start` or `stop` is not positive, or if `step` is not positive.
        """
        if start <= 0 or stop <= 0:
            raise ValueError(f"`start` and `stop` must be positive c/a ratios, got start={start}, stop={stop}.")
        if step <= 0:
            raise ValueError(f"`step` must be positive, got step={step}.")

        self.c_a_ratios: np.ndarray = np.arange(start=start, stop=stop, step=step)
        self.symprec = symprec

    def apply_transformation(
        self,
        structure: Structure,
    ) -> dict[float, Structure]:
        """Generate displaced structures along the Bain path.

        The Bain strain is defined relative to the Cartesian x/y/z axes, so `structure` is first
        standardized to a conventional, axis-aligned cubic cell (via `SpacegroupAnalyzer`) before
        being deformed.

        Args:
            structure (Structure): The input structure to be displaced along the Bain path.

        Returns:
            dict[float, Structure]: The displaced structures, keyed by the corresponding c/a ratio.
        """
        conventional_structure = SpacegroupAnalyzer(structure, symprec=self.symprec).get_conventional_standard_structure()

        displaced_structures: dict[float, Structure] = {}
        for c_a in self.c_a_ratios:
            delta = np.cbrt(1 / c_a) - 1
            displaced_structures[c_a] = self._get_displaced_structures(delta, conventional_structure)
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
            [1.0 + delta, 0.0, 0.0],
            [0.0, 1.0 + delta, 0.0],
            [0.0, 0.0, 1.0 / (1.0 + delta) ** 2.0],
        ]

        deformation = DeformStructureTransformation(deformation=transformation_matrix)
        return deformation.apply_transformation(structure)
