"""This module provides a class to calculate the cubic elastic constants of a given structure.

The `CubicElasticConstantsAnalyzer` class computes the elastic constants (C11, C12, and C44)
for a cubic crystal structure using energy-volume data and various deformation modes. The class
also computes additional mechanical properties such as bulk modulus, shear modulus, Poisson's ratio,
and Pugh's ratio based on the calculated elastic constants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.analysis.eos import EOS

from materialsframework.analysis.base import BaseAnalyzer
from materialsframework.analysis.utils import require_properties
from materialsframework.transformations.cubic_elastic_constants import (
    CubicElasticConstantsDeformationTransformation,
)
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from pymatgen.core import Structure

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

EV_A3_TO_GPA: float = 160.21766208


class CubicElasticConstantsAnalyzer(BaseAnalyzer):
    """A class used to calculate cubic elastic constants for a given structure.

    Computes the elastic constants (C11, C12, C44) for a cubic crystal structure from deformation and
    energy-volume data, along with derived mechanical properties such as bulk modulus, shear modulus,
    Young's modulus, and Poisson's ratio.
    """

    def __init__(
        self,
        eos_name: Literal[
            "murnaghan", "birch", "birch_murnaghan", "pourier_tarantola", "vinet", "deltafactor", "numerical_eos"
        ] = "birch_murnaghan",
        delta_max: float = 0.05,
        step_size: float = 0.01,
        calculator: BaseCalculator | None = None,
        cubic_transformation: CubicElasticConstantsDeformationTransformation | None = None,
    ) -> None:
        """Initializes the `CubicElasticConstantsAnalyzer` object.

        Args:
            eos_name (str, optional): The name of the equation of state (EOS) used for fitting energy-volume data.
                Defaults to "birch_murnaghan".
            delta_max (float, optional): The maximum deformation magnitude. Defaults to 0.05.
            step_size (float, optional): The step size for deformations. Defaults to 0.01.
            calculator (BaseCalculator | None, optional): The calculator object used for energy calculations.
            cubic_transformation (CubicElasticConstantsDeformationTransformation | None, optional): The transformation
                object used to apply cubic distortions.
        """
        super().__init__(calculator)
        self.eos_name = eos_name
        self.delta_max = delta_max
        self.step_size = step_size

        self._cubic_transformation = cubic_transformation

    @require_properties("energy")
    def calculate(self, structure: Structure | Atoms, is_relaxed: bool = False) -> dict[str, float]:
        """Calculates the cubic elastic constants for a given structure.

        Applies cubic distortions to the input structure, computes the potential energies of the deformed
        structures, and derives the elastic constants (C11, C12, C44) and additional mechanical properties
        from the resulting energy differences.

        Args:
            structure (Structure | Atoms): The input structure.
            is_relaxed (bool, optional): Whether the structure is already relaxed. Defaults to False.

        Returns:
            dict[str, float]: Dictionary with keys:
                - ``C11``: Elastic constant C11 in GPa.
                - ``C12``: Elastic constant C12 in GPa.
                - ``C44``: Elastic constant C44 in GPa.
                - ``youngs_modulus``: Young's modulus in GPa.
                - ``voigt_bulk_modulus``: Voigt bulk modulus in GPa.
                - ``voigt_shear_modulus``: Voigt shear modulus in GPa.
                - ``reuss_bulk_modulus``: Reuss bulk modulus in GPa.
                - ``reuss_shear_modulus``: Reuss shear modulus in GPa.
                - ``voigt_reuss_hill_bulk_modulus``: Voigt-Reuss-Hill bulk modulus in GPa.
                - ``voigt_reuss_hill_shear_modulus``: Voigt-Reuss-Hill shear modulus in GPa.
                - ``poisson_ratio``: Poisson ratio.
                - ``pugh_ratio``: Pugh ratio (G_VRH / K_VRH).
                - ``chen_vickers_hardness``: Chen-Vickers hardness in GPa.

        Raises:
            ValueError: If the calculator object does not have the 'energy' property implemented.
        """
        structure = self._ensure_relaxed(structure, is_relaxed)

        initial_volume: float = structure.volume

        distorted = self.cubic_transformation.apply_transformation(structure=structure)
        reference_energy: float = self.calculator.calculate(structure=structure)["energy"]

        bulk_modulus = self._get_bulk_modulus(distorted["uniform"], reference_energy)
        tetragonal_shear_modulus = self._get_tetragonal_shear_modulus(distorted["orthorhombic"], initial_volume, reference_energy)
        shear_modulus = self._get_shear_modulus(distorted["monoclinic"], initial_volume, reference_energy)

        c11 = bulk_modulus + (4 / 3 * tetragonal_shear_modulus)
        c12 = bulk_modulus - (2 / 3 * tetragonal_shear_modulus)
        c44 = shear_modulus

        elastic_tensor = self._build_cubic_elastic_tensor(c11, c12, c44)
        pugh_ratio = elastic_tensor.g_vrh / elastic_tensor.k_vrh
        chen_vickers_hardness = 2.0 * (pugh_ratio**2 * elastic_tensor.g_vrh) ** 0.585 - 3.0

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
            "pugh_ratio": pugh_ratio,
            "chen_vickers_hardness": chen_vickers_hardness,
        }

    @lazy_property("_cubic_transformation")
    def cubic_transformation(self) -> CubicElasticConstantsDeformationTransformation:
        """Returns the cubic transformation object used to generate deformed structures.

        Returns:
            CubicElasticConstantsDeformationTransformation: The transformation object used for cubic distortions.
        """
        return CubicElasticConstantsDeformationTransformation(delta_max=self.delta_max, step_size=self.step_size)

    def _fit_eos(self, volumes: Sequence[float], energies: Sequence[float]) -> float:
        """Fits the equation of state (EOS) to the given volumes and energies, returning the bulk modulus.

        Args:
            volumes (Sequence[float]): A sequence of volumes.
            energies (Sequence[float]): A sequence of energies.

        Returns:
            float: The bulk modulus obtained from the EOS fit in GPa.
        """
        eos_fit = EOS(eos_name=self.eos_name).fit(volumes=volumes, energies=energies)
        return eos_fit.b0_GPa

    @staticmethod
    def _fit_poly(deltas: Sequence[float], energies: Sequence[float], degree: int = 2) -> float:
        """Fits a polynomial to the given deltas and energies data points and calculates the second-order coefficient.

        Args:
            deltas (Sequence[float]): The sequence of delta values.
            energies (Sequence[float]): The sequence of energy values.
            degree (int, optional): The degree of the polynomial to fit. Defaults to 2.

        Returns:
            float: The second-order coefficient of the polynomial fit.
        """
        fit_coefficients = np.polynomial.polynomial.polyfit(deltas, energies, degree)
        return fit_coefficients[2]

    def _get_bulk_modulus(self, uniform_distorted_structures: dict[float, Structure], reference_energy: float) -> float:
        """Calculates the bulk modulus using equation of state (EOS) fitting.

        Args:
            uniform_distorted_structures (dict[float, Structure]): Dictionary mapping delta values to uniformly
                distorted structures.
            reference_energy (float): The energy of the undeformed (delta=0) structure, reused instead of
                recomputing it for the delta=0 entry.

        Returns:
            float: The bulk modulus in GPa.
        """
        volumes = [deformed_structure.volume for deformed_structure in uniform_distorted_structures.values()]
        energies = [
            reference_energy if delta == 0 else self.calculator.calculate(structure=deformed_structure)["energy"]
            for delta, deformed_structure in uniform_distorted_structures.items()
        ]
        return self._fit_eos(volumes, energies)

    def _get_tetragonal_shear_modulus(
        self,
        orthorhombic_distorted_structures: dict[float, Structure],
        initial_volume: float,
        reference_energy: float,
    ) -> float:
        """Calculates the tetragonal shear modulus from orthorhombic distortions.

        Args:
            orthorhombic_distorted_structures (dict[float, Structure]): Dictionary mapping delta values to
                orthorhombically distorted structures.
            initial_volume (float): The initial volume of the undeformed structure.
            reference_energy (float): The energy of the undeformed (delta=0) structure, reused instead of
                recomputing it for the delta=0 entry.

        Returns:
            float: The tetragonal shear modulus in GPa.
        """
        deltas = list(orthorhombic_distorted_structures.keys())
        energies = [
            reference_energy if delta == 0 else self.calculator.calculate(structure=deformed_structure)["energy"]
            for delta, deformed_structure in orthorhombic_distorted_structures.items()
        ]
        return EV_A3_TO_GPA * (self._fit_poly(deltas, energies) / (2 * initial_volume))

    def _get_shear_modulus(
        self,
        monoclinic_distorted_structures: dict[float, Structure],
        initial_volume: float,
        reference_energy: float,
    ) -> float:
        """Calculates the shear modulus from monoclinic distortions.

        Args:
            monoclinic_distorted_structures (dict[float, Structure]): Dictionary mapping delta values to
                monoclinically distorted structures.
            initial_volume (float): The initial volume of the undeformed structure.
            reference_energy (float): The energy of the undeformed (delta=0) structure, reused instead of
                recomputing it for the delta=0 entry.

        Returns:
            float: The shear modulus in GPa.
        """
        deltas = list(monoclinic_distorted_structures.keys())
        energies = [
            reference_energy if delta == 0 else self.calculator.calculate(structure=deformed_structure)["energy"]
            for delta, deformed_structure in monoclinic_distorted_structures.items()
        ]
        return EV_A3_TO_GPA * (self._fit_poly(deltas, energies) / (2 * initial_volume))

    @staticmethod
    def _build_cubic_elastic_tensor(c11: float, c12: float, c44: float) -> ElasticTensor:
        """Builds the 6x6 cubic elastic tensor from the given elastic constants.

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
