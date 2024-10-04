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

from materialsframework.calculators.m3gnet import M3GNetCalculator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.tools.calculator import BaseCalculator

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
            calculator: BaseCalculator | None = None,
    ) -> None:
        """
        Initializes the `BainDisplacementTransformation` object.

        Args:
            start (float, optional): The starting displacement value for the c/a ratio. Defaults to 0.89.
            stop (float, optional): The stopping displacement value for the c/a ratio. Defaults to 1.4.
            step (float, optional): The step size for incrementing the c/a ratio. Defaults to 0.01.
            calculator (BaseCalculator | None, optional): A calculator object for structure relaxation.
                                                             If None, defaults to `M3GNetCalculator`.
        """
        self._calculator = calculator

        self.c_a_ratios: np.ndarray = np.arange(start=start, stop=stop, step=step)
        self.displaced_structures: dict[float, Structure] = {}

    def apply_transformation(
            self,
            structure: Structure,
            is_relaxed: bool = False
    ) -> None:
        """
        Applies the Bain displacement transformation to generate structures along the Bain path.

        This method generates displaced structures for each value of the c/a ratio in the specified range.
        If the `is_relaxed` flag is set to False, the method relaxes the input structure before applying
        the Bain path displacement. The resulting structures are stored in the `displaced_structures` dictionary.

        Args:
            structure (Structure): The input structure to be displaced along the Bain path.
            is_relaxed (bool, optional): If False, the input structure is first relaxed before applying the transformation.
                                         Defaults to False.

        Note:
            The generated structures are stored in the `displaced_structures` attribute, keyed by the corresponding c/a ratio.
        """
        if not is_relaxed:
            structure: Structure = self._relax_structure(structure)  # type: ignore

        for c_a in self.c_a_ratios:
            delta = np.cbrt(1 / c_a) - 1
            self.displaced_structures[c_a] = self._get_displaced_structures(delta, structure)

    @property
    def calculator(self) -> BaseCalculator:
        """
        Returns the Calculator instance for structure relaxation.

        If the calculator instance is not already created, it creates a new `M3GNetCalculator` instance
        and returns it. Otherwise, it returns the existing calculator instance.

        Returns:
            BaseCalculator: The calculator instance used for structure relaxation.
        """
        if self._calculator is None:
            self._calculator = M3GNetCalculator()
        return self._calculator

    def _relax_structure(
            self,
            structure: Structure
    ) -> Structure:
        """
        Relaxes the input structure using the calculator.

        This method takes a pymatgen `Structure` object as input and relaxes it using the specified calculator.
        The relaxed structure is returned.

        Args:
            structure (Structure): The initial structure to be relaxed.

        Returns:
            Structure: The relaxed structure.
        """
        return self.calculator.relax(structure)["final_structure"]

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
