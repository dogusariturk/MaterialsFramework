"""
This module contains a class to calculate the cubic elastic constants of a given relaxed structure.
"""
from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import numpy as np
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.analysis.eos import EOS

from materialsframework.calculators.m3gnet import M3GNetCalculator
from materialsframework.transformations.elastic_constants import CubicElasticConstantsDeformationTransformation

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.calculators.typing import Calculator

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

eV_A3_to_GPa: float = 160.21766208


class CubicElasticConstantsAnalyzer:
    """
    A class used to represent a Cubic Elastic Constants Analyzer.

    This class provides methods to calculate the cubic elastic constants for a given relaxed structure.
    """

    def __init__(
            self,
            eos_name: str = "birch_murnaghan",
            calculator: Optional[Calculator] = None,
            cubic_transformation: Optional[CubicElasticConstantsDeformationTransformation] = None
    ) -> None:
        """
        Initializes the CubicElasticConstantsAnalyzer.

        Parameters:
            eos_name (str): The name of the equation of state (EOS) to use for fitting energy-volume data.
                            Default is "birch_murnaghan".
            calculator (Optional[Calculator]): The calculator object to use for calculating potential energies.
            cubic_transformation (Optional[CubicElasticConstantsDeformationTransformation]): The cubic transformation object.
        """
        self._eos_name = eos_name

        self._calculator = calculator
        self._cubic_transformation = cubic_transformation

    def calculate(self, undeformed_structure: Structure, is_relaxed: bool = False) -> dict:
        """
        Calculates the cubic elastic constants for the given undeformed structure.

        Parameters:
            undeformed_structure (Structure): The undeformed relaxed structure.
            is_relaxed (bool): Whether the undeformed structure is already relaxed. Default is False.

        Returns:
            dict: A dictionary containing the calculated cubic elastic constants.
        """
        initial_volume: float = undeformed_structure.volume

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
    def calculator(self) -> Calculator:
        """
        Gets the calculator used for calculating potential energies.
        If not set, initializes a new M3GNetCalculator.

        Returns:
            Calculator: The calculator object.
        """
        if self._calculator is None:
            self._calculator = M3GNetCalculator()
        return self._calculator

    @property
    def cubic_transformation(self) -> CubicElasticConstantsDeformationTransformation:
        """
        Gets the cubic transformation object.

        Returns:
            CubicElasticConstantsDeformationTransformation: The cubic transformation object.
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
        Fits the equation of state (EOS) to the given volumes and energies
        and returns the bulk modulus.

        Parameters:
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
        Fits a polynomial curve to the given deltas and energies data points
        and calculates the second order coefficient.

        Parameters:
            deltas (list[float, ...]): The array of delta values.
            energies (list[float, ...]): The array of energy values.
            degree (int): The degree of the polynomial to fit. Default is 2.

        Returns:
            float: The calculated second order coefficient.
        """
        fit_coefficients = np.polynomial.polynomial.polyfit(deltas, energies, degree)
        return fit_coefficients[2]

    def _get_bulk_modulus(self) -> float:
        """
        Calculates the bulk modulus using the equation of state (EOS) fitting.

        Returns:
            float: The calculated bulk modulus.
        """
        volumes, energies = zip(
                *[(deformed_structure.volume,
                   self.calculator.calculate(structure=deformed_structure)["potential_energy"],) for
                  _, deformed_structure in self.cubic_transformation.uniform_distorted_structures.items()])
        return self._fit_eos(volumes, energies)

    def _get_tetragonal_shear_modulus(self, initial_volume: float) -> float:
        """
        Calculates the tetragonal shear modulus.

        Args:
            initial_volume (float): The initial volume of the undeformed structure.

        Returns:
            float: The calculated tetragonal shear modulus.
        """
        deltas, energies = zip(
                *[(delta, self.calculator.calculate(structure=deformed_structure)["potential_energy"],) for
                  delta, deformed_structure in self.cubic_transformation.orthorhombic_distorted_structures.items()])
        return eV_A3_to_GPa * (self._fit_poly(deltas, energies) / (2 * initial_volume))

    def _get_shear_modulus(self, initial_volume: float) -> float:
        """
        Calculates the shear modulus using the given calculator and cubic transformation.

        Args:
            initial_volume (float): The initial volume of the undeformed structure.

        Returns:
            float: The shear modulus calculated from the potential energy and deformation values.
        """
        deltas, energies = zip(
                *[(delta, self.calculator.calculate(structure=deformed_structure)["potential_energy"],) for
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
            ElasticTensor: pymatgen ElasticTensor object.
        """
        elastic_tensor = np.zeros([6, 6])
        elastic_tensor[:3, :3].fill(c12)
        np.fill_diagonal(elastic_tensor, [c11] * 3 + [c44] * 3)
        return ElasticTensor.from_voigt(elastic_tensor)
