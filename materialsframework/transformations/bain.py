"""
This module provides a class to generate displaced structures along the Bain Path.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.transformations.standard_transformations import DeformStructureTransformation

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.calculators import Relaxer

from materialsframework.calculators import M3GNetRelaxer

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class BainDisplacementTransformation:
    """
    A class used to represent a Bain Displacement Transformation.

    This class provides a class to generate displaced structures along the Bain Path.
    """

    def __init__(
            self,
            start: float = 0.89,
            stop: float = 1.4,
            step: float = 0.01,
            fmax: float = 1e-5,
            relax_cell: bool = True,
            verbose: bool = False,
            steps: int = 1000,
            model: str = "M3GNet-MP-2021.2.8-PES"
    ) -> None:
        """
        Initializes the BainDisplacementTransformation.

        Args:
            start (float): The starting displacement value. Defaults to 0.89.
            stop (float): The stopping displacement value. Defaults to 1.4.
            step (float): The step size for the displacement values. Defaults to 0.01.
            fmax (float): The maximum force tolerance for relaxation. Defaults to 1e-5.
            relax_cell (bool): Whether to relax the lattice cell. Defaults to True.
            verbose (bool): Whether to print verbose output. Defaults to False.
            steps (int): The maximum number of relaxation steps. Defaults to 1000.
            model (str): The potential model to use for relaxation. Defaults to "M3GNet-MP-2021.2.8-PES".
        """
        self._fmax = fmax
        self._relax_cell = relax_cell
        self._verbose = verbose
        self._steps = steps
        self._model = model

        self._relaxer = None

        self.c_a_ratios: np.ndarray = np.arange(start=start, stop=stop, step=step)
        self.displaced_structures: dict[float, Structure] = {}

    def apply_transformation(
            self,
            structure: Structure,
            is_relaxed: bool = False
    ) -> None:
        """
        Applies the transformation to generate displaced structures along the Bain path.
        """
        if not is_relaxed:
            structure: Structure = self._relax_structure(structure)  # type: ignore

        for c_a in self.c_a_ratios:
            delta = np.cbrt(1 / c_a) - 1
            self.displaced_structures[c_a] = self._get_displaced_structures(delta, structure)

    @property
    def relaxer(self) -> Relaxer:
        """
        Returns the Relaxer instance.

        If the relaxer instance is not already created, it creates a new M3GNetRelaxer instance
        and returns it. Otherwise, it returns the existing relaxer instance.

        Returns:
            Relaxer: The Relaxer instance.
        """
        if self._relaxer is None:
            self._relaxer = M3GNetRelaxer(fmax=self._fmax,
                                          relax_cell=self._relax_cell,
                                          verbose=self._verbose,
                                          steps=self._steps,
                                          model=self._model)
        return self._relaxer

    def _relax_structure(self, structure: Structure) -> Structure:
        """
        This method takes a pymatgen Structure object as input and returns a relaxed structure.
        The relaxation is performed using the M3GNetRelaxer instance associated with the class.

        Args:
            structure (Structure): The initial pymatgen Structure object that needs to be relaxed.

        Returns:
            Structure: The relaxed pymatgen Structure object.
        """
        return self.relaxer.relax(structure)['final_structure']

    @staticmethod
    def _get_displaced_structures(
            delta: float,
            structure: Structure
    ) -> Structure:
        """
        Applies the transformation to generate a displaced structure.

        Args:
            delta (float): The displacement value.
            structure (Structure): The input structure to be displaced.

        Returns:
            Structure: The displaced structure.
        """
        transformation_matrix = ([[1 + delta, 0, 0],
                                  [0, 1 + delta, 0],
                                  [0, 0, 1 / (1 + delta) ** 2]])

        deformation = DeformStructureTransformation(deformation=transformation_matrix)
        return deformation.apply_transformation(structure)
