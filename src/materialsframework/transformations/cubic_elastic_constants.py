"""Generates distorted structures for elastic constant calculations in cubic systems.

Applies uniform, orthorhombic, and monoclinic distortions used to compute the corresponding
elastic moduli.
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


class CubicElasticConstantsDeformationTransformation:
    """Generates deformed structures for cubic elastic constant calculations.

    Distortion magnitudes are set by a range of delta values, from ``-delta_max`` to ``delta_max``
    in steps of ``step_size``.
    """

    def __init__(
        self,
        delta_max: float = 0.05,
        step_size: float = 0.01,
    ) -> None:
        """Initializes the `CubicElasticConstantsDeformationTransformation` object.

        Args:
            delta_max (float, optional): The maximum delta value for the distortions. Defaults to 0.05.
            step_size (float, optional): The step size for the delta values. Defaults to 0.01.
        """
        self.delta_max = delta_max
        self.step_size = step_size

        self.deltas: np.ndarray = np.linspace(
            start=-1 * self.delta_max,
            stop=self.delta_max,
            num=int(2 * self.delta_max / self.step_size) + 1,
        )

    def apply_transformation(
        self,
        structure: Structure,
    ) -> dict[str, dict[float, Structure]]:
        """Generate distorted structures for elastic constant calculations.

        Args:
            structure (Structure): The input structure to be distorted.

        Returns:
            dict[str, dict[float, Structure]]: Dictionary with keys:
                - ``uniform``: Dictionary mapping delta values to uniformly distorted structures.
                - ``orthorhombic``: Dictionary mapping delta values to orthorhombically distorted structures.
                - ``monoclinic``: Dictionary mapping delta values to monoclinically distorted structures.
        """
        uniform: dict[float, Structure] = {}
        orthorhombic: dict[float, Structure] = {}
        monoclinic: dict[float, Structure] = {}

        for delta in self.deltas:
            uniform[delta] = self._apply_uniform_distortion(delta, structure)
            if delta >= 0:
                orthorhombic[delta] = self._apply_orthorhombic_distortion(delta, structure)
                monoclinic[delta] = self._apply_monoclinic_distortion(delta, structure)

        return {"uniform": uniform, "orthorhombic": orthorhombic, "monoclinic": monoclinic}

    def _apply_monoclinic_distortion(self, delta: float, structure: Structure) -> Structure:
        """Apply a monoclinic distortion to the structure.

        Args:
            delta (float): The magnitude of the monoclinic distortion.
            structure (Structure): The input structure to be distorted.

        Returns:
            Structure: The monoclinically distorted structure.
        """
        _monoclinic_distortion = (
            [1, delta, 0],
            [delta, 1, 0],
            [0, 0, 1 / (1 - delta**2)],
        )
        return self._apply_deformation(structure=structure, deformation=_monoclinic_distortion)

    def _apply_orthorhombic_distortion(self, delta: float, structure: Structure) -> Structure:
        """Apply an orthorhombic distortion to the structure.

        Args:
            delta (float): The magnitude of the orthorhombic distortion.
            structure (Structure): The input structure to be distorted.

        Returns:
            Structure: The orthorhombically distorted structure.
        """
        _orthorhombic_distortion = (
            [1 + delta, 0, 0],
            [0, 1 - delta, 0],
            [0, 0, 1 / (1 - delta**2)],
        )
        return self._apply_deformation(structure=structure, deformation=_orthorhombic_distortion)

    def _apply_uniform_distortion(self, delta: float, structure: Structure) -> Structure:
        """Apply a uniform distortion to the structure.

        Args:
            delta (float): The magnitude of the uniform distortion.
            structure (Structure): The input structure to be distorted.

        Returns:
            Structure: The uniformly distorted structure.
        """
        _uniform_distortion = ([1 + delta, 0, 0], [0, 1 + delta, 0], [0, 0, 1 + delta])
        return self._apply_deformation(structure=structure, deformation=_uniform_distortion)

    @staticmethod
    def _apply_deformation(structure: Structure, deformation: tuple) -> Structure:
        """Apply the given deformation matrix to the structure.

        Args:
            structure (Structure): The input structure to be deformed.
            deformation (tuple): The deformation matrix to be applied.

        Returns:
            Structure: The deformed structure.
        """
        return DeformStructureTransformation(deformation).apply_transformation(structure)
