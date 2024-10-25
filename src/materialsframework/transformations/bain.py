"""
This module provides a class to generate displaced structures along the Bain Path.

The `BainDisplacementTransformation` class allows users to generate structures that are deformed
along the Bain path, which describes the transition between body-centered cubic (BCC) and face-centered
cubic (FCC) crystal structures through a continuous deformation of the lattice.
This transformation is useful in studying martensitic transformations and phase transitions in materials.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymatgen.transformations.standard_transformations import DeformStructureTransformation

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class BainDisplacementTransformation:
    """
    A class used to generate displaced structures along the Bain Path.

    The `BainDisplacementTransformation` class provides methods to generate a series of deformed structures
    by varying the c/a ratio in small steps, following the Bain transformation pathway. This pathway is significant
    for studying phase transformations between cubic and tetragonal structures, especially in materials that undergo
    martensitic transformations.
    """

    def __init__(
            self,
            start: float = 0.89,
            stop: float = 1.4,
            step: float = 0.01,
    ) -> None:
        """
        Initializes the `BainDisplacementTransformation` object.

        Args:
            start (float, optional): The starting displacement value for the c/a ratio. Defaults to 0.89.
            stop (float, optional): The stopping displacement value for the c/a ratio. Defaults to 1.4.
            step (float, optional): The step size for incrementing the c/a ratio. Defaults to 0.01.
        """
        self.c_a_ratios: np.ndarray = np.arange(start=start, stop=stop, step=step)
        self.displaced_structures: dict[float, Structure] = {}

    def apply_transformation(
            self,
            structure: Structure,
    ) -> None:
        """
        Applies the Bain displacement transformation to generate structures along the Bain path.

        This method generates displaced structures for each value of the c/a ratio in the specified range.
        If the `is_relaxed` flag is set to False, the method relaxes the input structure before applying
        the Bain path displacement. The resulting structures are stored in the `displaced_structures` dictionary.

        Args:
            structure (Structure): The input structure to be displaced along the Bain path.

        Note:
            The generated structures are stored in the `displaced_structures` attribute, keyed by the corresponding c/a ratio.
        """
        for c_a in self.c_a_ratios:
            delta = np.cbrt(1 / c_a) - 1
            self.displaced_structures[c_a] = self._get_displaced_structures(delta, structure)

    @staticmethod
    def _get_displaced_structures(
            delta: float,
            structure: Structure
    ) -> Structure:
        """
        Generates a displaced structure by applying a deformation along the Bain path.

        The deformation is applied based on the given `delta` value, which modifies the lattice parameters
        in accordance with the Bain transformation. A new structure is returned with the deformation applied.

        Args:
            delta (float): The displacement value for the Bain transformation.
            structure (Structure): The input structure to be deformed.

        Returns:
            Structure: The displaced structure with the Bain transformation applied.
        """
        transformation_matrix = ([[1 + delta, 0, 0],
                                  [0, 1 + delta, 0],
                                  [0, 0, 1 / (1 + delta) ** 2]])

        deformation = DeformStructureTransformation(deformation=transformation_matrix)
        return deformation.apply_transformation(structure)
