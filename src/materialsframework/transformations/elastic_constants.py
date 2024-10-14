"""
This module provides a class to generate distorted structures for elastic constant calculations.

The `CubicElasticConstantsDeformationTransformation` class facilitates the generation of distorted
structures required for the calculation of elastic constants in cubic systems. It supports the application
of uniform, orthorhombic, and monoclinic distortions, which are used in the calculation of the corresponding
elastic moduli.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymatgen.transformations.standard_transformations import DeformStructureTransformation

if TYPE_CHECKING:
    from pymatgen.core import Structure

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
    ) -> None:
        """
        Initializes the `CubicElasticConstantsDeformationTransformation` object.

        Args:
            delta_max (float, optional): The maximum delta value for the distortions. Defaults to 0.05.
            step_size (float, optional): The step size for the delta values. Defaults to 0.01.
        """
        self.delta_max = delta_max
        self.step_size = step_size

        self.deltas: np.ndarray = np.linspace(start=-1 * self.delta_max,
                                              stop=self.delta_max,
                                              num=int(2 * self.delta_max / self.step_size) + 1)

        self.uniform_distorted_structures: dict[float, Structure] = {}
        self.orthorhombic_distorted_structures: dict[float, Structure] = {}
        self.monoclinic_distorted_structures: dict[float, Structure] = {}

    def apply_transformation(
            self,
            structure: Structure,
    ) -> None:
        """
        Applies the transformation to generate distorted structures for elastic constant calculations.

        This method generates distorted structures for each delta value in the specified range. The resulting
        structures are stored in the class attributes for further elastic constant analysis.

        Args:
            structure (Structure): The input structure to be distorted.

        Note:
            The distorted structures are stored in dictionaries under keys corresponding to the delta value.
        """
        for delta in self.deltas:
            self._apply_uniform_distortion(delta, structure)
            if delta >= 0:
                self._apply_orthorhombic_distortion(delta, structure)
                self._apply_monoclinic_distortion(delta, structure)

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
