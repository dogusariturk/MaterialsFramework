"""
This module contains a class to calculate the cubic elastic constants of a given relaxed structure.

The `CubicElasticConstantsAnalyzer` class computes the elastic constants (C11, C12, and C44)
for a cubic crystal structure using energy-volume data and various deformation modes. The class
also computes additional mechanical properties such as bulk modulus, shear modulus, Poisson's ratio,
and Pugh's ratio based on the calculated elastic constants.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.analysis.eos import EOS

from materialsframework.calculators.m3gnet import M3GNetCalculator
from materialsframework.transformations.elastic_constants import CubicElasticConstantsDeformationTransformation

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

eV_A3_to_GPa: float = 160.21766208


class CubicElasticConstantsAnalyzer:
    """
    A class used to calculate cubic elastic constants for a given relaxed structure.

    The `CubicElasticConstantsAnalyzer` class provides methods to compute the elastic constants
    (C11, C12, C44) for a cubic crystal structure using deformation and energy-volume data. In addition
    to the elastic constants, this class computes mechanical properties such as bulk modulus, shear modulus,
    Young's modulus, and Poisson's ratio.
    """

    def __init__(
            self,
            eos_name: str = "birch_murnaghan",
            calculator: BaseCalculator | None = None,
            cubic_transformation: CubicElasticConstantsDeformationTransformation | None = None
    ) -> None:
        """
        Initializes the `CubicElasticConstantsAnalyzer` object.

        Args:
            eos_name (str, optional): The name of the equation of state (EOS) used for fitting energy-volume data.
                                      Defaults to "birch_murnaghan".
            calculator (BaseCalculator | None, optional): The calculator object used for energy calculations.
                                                          Defaults to `M3GNetCalculator`.
            cubic_transformation (CubicElasticConstantsDeformationTransformation | None, optional): The transformation
                                                                                                    object used to apply
                                                                                                    cubic distortions.
        """
        self._eos_name = eos_name

        self._calculator = calculator
        self._cubic_transformation = cubic_transformation

    def calculate(
            self,
            undeformed_structure: Structure,
            is_relaxed: bool = False
    ) -> dict[str, float]:
        """
        Calculates the cubic elastic constants for a given undeformed structure.

        This method applies cubic distortions to the input structure and computes the potential energies
        of the deformed structures. The elastic constants (C11, C12, C44) are calculated based on these
        energy differences, and additional mechanical properties are computed.

        Args:
            undeformed_structure (Structure): The undeformed, relaxed structure.
            is_relaxed (bool, optional): Whether the structure is already relaxed. Defaults to False.

        Returns:
            dict[str, float]: A dictionary containing the calculated cubic elastic constants (C11, C12, C44) and
                              various derived mechanical properties.
        """
        if "energy" not in self.calculator.AVAILABLE_PROPERTIES:
            raise ValueError("The calculator object must have the 'energy' property implemented.")

        initial_volume: float = undeformed_structure.volume  # FIXME: This volume is before relaxation!

        self.cubic_transformation.apply_transformation(structure=undeformed_structure,
                                                       is_relaxed=is_relaxed)

        bulk_modulus = self._get_bulk_modulus()
        tetragonal_shear_modulus = self._get_tetragonal_shear_modulus(initial_volume)
        shear_modulus = self._get_shear_modulus(initial_volume)

        c11 = bulk_modulus + (4 / 3 * tetragonal_shear_modulus)
        c12 = bulk_modulus - (2 / 3 * tetragonal_shear_modulus)
        c44 = shear_modulus

        elastic_tensor = self._build_cubic_elastic_tensor(c11, c12, c44)

        return {
                "C11": c11,
                "C12": c12,
                "C44": c44,
                "youngs_modulus": elastic_tensor.y_mod / 1e9,
                "voigt_bulk_modulus": elastic_tensor.k_voigt,
                "voigt_shear_modulus": elastic_tensor.g_voigt,
                "reuss_bulk_modulus": elastic_tensor.k_reuss,
                "reuss_shear_modulus": elastic_tensor.g_reuss,
                "voigt_reuss_hill_bulk_modulus": elastic_tensor.k_vrh,
                "voigt_reuss_hill_shear_modulus": elastic_tensor.g_vrh,
                "poisson_ratio": elastic_tensor.homogeneous_poisson,
                "pugh_ratio": elastic_tensor.g_vrh / elastic_tensor.k_vrh,
        }

    @property
    def calculator(self) -> BaseCalculator:
        """
        Returns the calculator instance used for energy calculations.

        If the calculator instance is not already initialized, this method creates a new `M3GNetCalculator` instance.

        Returns:
            BaseCalculator: The calculator object used for energy calculations.
        """
        if self._calculator is None:
            self._calculator = M3GNetCalculator()
        return self._calculator

    @property
    def cubic_transformation(self) -> CubicElasticConstantsDeformationTransformation:
        """
        Returns the cubic transformation object used to generate deformed structures.

        If the transformation instance is not already initialized, this method creates a new `CubicElasticConstantsDeformationTransformation` instance.

        Returns:
            CubicElasticConstantsDeformationTransformation: The transformation object used for cubic distortions.
        """
        if self._cubic_transformation is None:
            self._cubic_transformation = CubicElasticConstantsDeformationTransformation()
        return self._cubic_transformation

    def _fit_eos(
            self,
            volumes: list[float, ...],
            energies: list[float, ...]
    ) -> float:
        """
        Fits the equation of state (EOS) to the given volumes and energies, returning the bulk modulus.

        Args:
            volumes (list[float, ...]): A list of volumes.
            energies (list[float, ...]): A list of energies.

        Returns:
            float: The bulk modulus obtained from the EOS fit in GPa.
        """
        eos_fit = EOS(eos_name=self._eos_name).fit(volumes=volumes, energies=energies)
        return eos_fit.b0_GPa

    @staticmethod
    def _fit_poly(
            deltas: list[float, ...],
            energies: list[float, ...],
            degree: int = 2
    ) -> float:
        """
        Fits a polynomial to the given deltas and energies data points and calculates the second-order coefficient.

        Args:
            deltas (list[float, ...]): The array of delta values.
            energies (list[float, ...]): The array of energy values.
            degree (int, optional): The degree of the polynomial to fit. Defaults to 2.

        Returns:
            float: The second-order coefficient of the polynomial fit.
        """
        fit_coefficients = np.polynomial.polynomial.polyfit(deltas, energies, degree)
        return fit_coefficients[2]

    def _get_bulk_modulus(self) -> float:
        """
        Calculates the bulk modulus using equation of state (EOS) fitting.

        Returns:
            float: The bulk modulus in GPa.
        """
        volumes, energies = zip(
                *[(deformed_structure.volume,
                   self.calculator.calculate(structure=deformed_structure)["energy"],) for
                  _, deformed_structure in self.cubic_transformation.uniform_distorted_structures.items()])
        return self._fit_eos(volumes, energies)

    def _get_tetragonal_shear_modulus(
            self,
            initial_volume: float
    ) -> float:
        """
        Calculates the tetragonal shear modulus from orthorhombic distortions.

        Args:
            initial_volume (float): The initial volume of the undeformed structure.

        Returns:
            float: The tetragonal shear modulus in GPa.
        """
        deltas, energies = zip(
                *[(delta, self.calculator.calculate(structure=deformed_structure)["energy"],) for
                  delta, deformed_structure in self.cubic_transformation.orthorhombic_distorted_structures.items()])
        return eV_A3_to_GPa * (self._fit_poly(deltas, energies) / (2 * initial_volume))

    def _get_shear_modulus(
            self,
            initial_volume: float
    ) -> float:
        """
        Calculates the shear modulus from monoclinic distortions.

        Args:
            initial_volume (float): The initial volume of the undeformed structure.

        Returns:
            float: The shear modulus in GPa.
        """
        deltas, energies = zip(
                *[(delta, self.calculator.calculate(structure=deformed_structure)["energy"],) for
                  delta, deformed_structure in self.cubic_transformation.monoclinic_distorted_structures.items()])
        return eV_A3_to_GPa * (self._fit_poly(deltas, energies) / (2 * initial_volume))

    @staticmethod
    def _build_cubic_elastic_tensor(
            c11: float,
            c12: float,
            c44: float
    ) -> ElasticTensor:
        """
        Builds the 6x6 cubic elastic tensor from the given elastic constants.

        Args:
            c11 (float): The C11 elastic constant.
            c12 (float): The C12 elastic constant.
            c44 (float): The C44 elastic constant.

        Returns:
            ElasticTensor: The pymatgen `ElasticTensor` object.
        """
        elastic_tensor = np.zeros([6, 6])
        elastic_tensor[:3, :3].fill(c12)
        np.fill_diagonal(elastic_tensor, [c11] * 3 + [c44] * 3)
        return ElasticTensor.from_voigt(elastic_tensor)
