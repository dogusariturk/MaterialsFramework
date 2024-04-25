"""
This module provides a class to generate distorted structures for elastic constant calculations.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.calculators import Relaxer

from materialsframework.calculators import M3GNetRelaxer

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class CubicElasticConstantsDeformationTransformation:
    """
    A class used to represent a Cubic Elastic Constants Transformation.

    This class provides methods to generate the deformed structures for the cubic elastic constant calculations.
    """

    def __init__(
            self,
            delta_max: float = 0.05,
            fmax: float = 1e-5,
            relax_cell: bool = True,
            verbose: bool = False,
            steps: int = 1000,
            model: str = "M3GNet-MP-2021.2.8-PES"
    ) -> None:
        """
        Initializes the CubicElasticConstantsDeformationTransformation.

        Args:
            delta_max (float): The maximum delta value for distortions. Defaults to 0.05.
            fmax (float): The maximum force tolerance for relaxation. Defaults to 1e-5.
            relax_cell (bool): Whether to relax the lattice cell. Defaults to True.
            verbose (bool): Whether to print verbose output during relaxation. Defaults to False.
            steps (int): The maximum number of relaxation steps. Defaults to 1000.
            model (str): The potential model to use for relaxation. Defaults to "M3GNet-MP-2021.2.8-PES".
        """
        self._fmax = fmax
        self._relax_cell = relax_cell
        self._verbose = verbose
        self._steps = steps
        self._model = model

        self._relaxer = None

        # TODO: The step size of 0.01 can be an input parameter
        self.deltas: np.ndarray = np.linspace(start=-1 * delta_max,
                                              stop=delta_max,
                                              num=int(2 * delta_max / 0.01) + 1)

        self.uniform_distorted_structures: dict[float, Structure] = {}
        self.orthorhombic_distorted_structures: dict[float, Structure] = {}
        self.monoclinic_distorted_structures: dict[float, Structure] = {}

    def apply_transformation(
            self,
            structure: Structure,
            is_relaxed: bool = False
    ) -> None:
        """
        Applies the transformation to generate distorted structures.

        Args:
            structure (Structure): The input structure to be distorted.
            is_relaxed (bool): Whether the input structure is already relaxed. Defaults to False.
        """
        if not is_relaxed:
            structure: Structure = self._relax_structure(structure)  # type: ignore

        for delta in self.deltas:
            self._apply_uniform_distortion(delta, structure)
            if delta >= 0:
                self._apply_orthorhombic_distortion(delta, structure)
                self._apply_monoclinic_distortion(delta, structure)

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

    def _apply_monoclinic_distortion(self, delta: float, structure: Structure) -> None:
        """
        Applies the monoclinic distortion to the structure.

        Args:
            delta (float): The delta value for the distortion.
            structure (Structure): The input structure to be distorted.
        """
        _monoclinic_distortion = ([1, delta, 0],
                                  [delta, 1, 0],
                                  [0, 0, 1 / (1 - delta ** 2)])
        self.monoclinic_distorted_structures[delta] = self._apply_deformation(
                structure=structure,
                deformation=_monoclinic_distortion
        )

    def _apply_orthorhombic_distortion(self, delta: float, structure: Structure) -> None:
        """
        Applies the orthorhombic distortion to the structure.

        Args:
            delta (float): The delta value for the distortion.
            structure (Structure): The input structure to be distorted.
        """
        _orthorhombic_distortion = ([1 + delta, 0, 0],
                                    [0, 1 - delta, 0],
                                    [0, 0, 1 / (1 - delta ** 2)])
        self.orthorhombic_distorted_structures[delta] = self._apply_deformation(
                structure=structure,
                deformation=_orthorhombic_distortion
        )

    def _apply_uniform_distortion(self, delta: float, structure: Structure) -> None:
        """
        Applies the uniform distortion to the structure.

        Args:
            delta (float): The delta value for the distortion.
            structure (Structure): The input structure to be distorted.
        """
        _uniform_distortion = ([1 + delta, 0, 0],
                               [0, 1 + delta, 0],
                               [0, 0, 1 + delta])
        self.uniform_distorted_structures[delta] = self._apply_deformation(
                structure=structure,
                deformation=_uniform_distortion
        )

    @staticmethod
    def _apply_deformation(structure: Structure, deformation: tuple) -> Structure:
        """
        Applies the deformation to the structure.

        Args:
            structure (Structure): The input structure to be deformed.
            deformation (tuple): The deformation matrix to be applied.

        Returns:
            Structure: The deformed structure.
        """
        return DeformStructureTransformation(deformation).apply_transformation(structure)
