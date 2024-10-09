"""
This module provides a class to generate distorted structures for elastic constant calculations.

The `CubicElasticConstantsDeformationTransformation` class facilitates the generation of distorted
structures required for the calculation of elastic constants in cubic systems. It supports the application
of uniform, orthorhombic, and monoclinic distortions, which are used in the calculation of the corresponding
elastic moduli.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.transformations.standard_transformations import DeformStructureTransformation

from materialsframework.calculators.m3gnet import M3GNetCalculator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.tools.calculator import BaseCalculator

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class CubicElasticConstantsDeformationTransformation:
    """
    A class used to generate deformed structures for cubic elastic constant calculations.

    The `CubicElasticConstantsDeformationTransformation` class provides methods to generate distorted
    structures for the calculation of elastic constants in cubic crystals. The distortions applied include
    uniform, orthorhombic, and monoclinic deformations. The deformation magnitudes are defined by a range
    of delta values, which control the level of distortion.
    """

    def __init__(
            self,
            delta_max: float = 0.05,
            step_size: float = 0.01,
            calculator: BaseCalculator | None = None,
    ) -> None:
        """
        Initializes the `CubicElasticConstantsDeformationTransformation` object.

        Args:
            delta_max (float, optional): The maximum delta value for the distortions. Defaults to 0.05.
            step_size (float, optional): The step size for the delta values. Defaults to 0.01.
            calculator (BaseCalculator | None, optional): A calculator object for structure relaxation.
                                                             If None, defaults to `M3GNetCalculator`.
        """
        self.delta_max = delta_max
        self.step_size = step_size

        self._calculator = calculator

        self.deltas: np.ndarray = np.linspace(start=-1 * self.delta_max,
                                              stop=self.delta_max,
                                              num=int(2 * self.delta_max / self.step_size) + 1)

        self.uniform_distorted_structures: dict[float, Structure] = {}
        self.orthorhombic_distorted_structures: dict[float, Structure] = {}
        self.monoclinic_distorted_structures: dict[float, Structure] = {}

    def apply_transformation(
            self,
            structure: Structure,
            is_relaxed: bool = False
    ) -> None:
        """
        Applies the transformation to generate distorted structures for elastic constant calculations.

        This method generates distorted structures for each delta value in the specified range. If the
        input structure is not relaxed, the method relaxes it before applying distortions. The resulting
        structures are stored in the class attributes for further elastic constant analysis.

        Args:
            structure (Structure): The input structure to be distorted.
            is_relaxed (bool, optional): Whether the input structure is already relaxed. Defaults to False.

        Note:
            The distorted structures are stored in dictionaries under keys corresponding to the delta value.
        """
        if not is_relaxed:
            structure: Structure = self._relax_structure(structure)  # type: ignore

        for delta in self.deltas:
            self._apply_uniform_distortion(delta, structure)
            if delta >= 0:
                self._apply_orthorhombic_distortion(delta, structure)
                self._apply_monoclinic_distortion(delta, structure)

    @property
    def calculator(self) -> BaseCalculator:
        """
        Returns the calculator instance used for structure relaxation.

        If the calculator instance is not already created, this method initializes a new `M3GNetCalculator`
        instance. Otherwise, it returns the existing calculator.

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

    def _apply_monoclinic_distortion(
            self,
            delta: float,
            structure: Structure
    ) -> None:
        """
        Applies a monoclinic distortion to the structure.

        This method generates a monoclinic deformation by modifying the lattice vectors according to the
        specified delta value. The distorted structure is stored in the `monoclinic_distorted_structures` attribute.

        Args:
            delta (float): The magnitude of the monoclinic distortion.
            structure (Structure): The input structure to be distorted.
        """
        _monoclinic_distortion = ([1, delta, 0],
                                  [delta, 1, 0],
                                  [0, 0, 1 / (1 - delta ** 2)])
        self.monoclinic_distorted_structures[delta] = self._apply_deformation(
                structure=structure,
                deformation=_monoclinic_distortion
        )

    def _apply_orthorhombic_distortion(
            self,
            delta: float,
            structure: Structure
    ) -> None:
        """
        Applies an orthorhombic distortion to the structure.

        This method generates an orthorhombic deformation by modifying the lattice vectors according to the
        specified delta value. The distorted structure is stored in the `orthorhombic_distorted_structures` attribute.

        Args:
            delta (float): The magnitude of the orthorhombic distortion.
            structure (Structure): The input structure to be distorted.
        """
        _orthorhombic_distortion = ([1 + delta, 0, 0],
                                    [0, 1 - delta, 0],
                                    [0, 0, 1 / (1 - delta ** 2)])
        self.orthorhombic_distorted_structures[delta] = self._apply_deformation(
                structure=structure,
                deformation=_orthorhombic_distortion
        )

    def _apply_uniform_distortion(
            self,
            delta: float,
            structure: Structure
    ) -> None:
        """
        Applies a uniform distortion to the structure.

        This method generates a uniform deformation by scaling all lattice vectors equally according to the
        specified delta value. The distorted structure is stored in the `uniform_distorted_structures` attribute.

        Args:
            delta (float): The magnitude of the uniform distortion.
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
    def _apply_deformation(
            structure: Structure,
            deformation: tuple
    ) -> Structure:
        """
        Applies the given deformation matrix to the structure.

        This method applies the specified deformation matrix to the input structure, returning the
        deformed structure.

        Args:
            structure (Structure): The input structure to be deformed.
            deformation (tuple): The deformation matrix to be applied.

        Returns:
            Structure: The deformed structure.
        """
        return DeformStructureTransformation(deformation).apply_transformation(structure)
