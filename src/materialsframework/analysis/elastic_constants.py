"""This module provides a class to calculate the elastic constants of a given structure.

The `ElasticConstantsAnalyzer` class fits a symmetry-appropriate elastic tensor from stress-strain
data, with optional fixed-cell ionic relaxation. It also derives polycrystalline moduli, the Debye
temperature, a Grüneisen parameter approximation, and elastic sound velocities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymatgen.analysis.elasticity import ElasticTensor

from materialsframework.analysis.base import BaseAnalyzer
from materialsframework.analysis.utils import require_properties
from materialsframework.constants import EV_A3_TO_GPA
from materialsframework.tools import elastic
from materialsframework.transformations.elastic_constants import (
    ElasticConstantsDeformationTransformation,
)
from materialsframework.utils import lazy_property, to_atoms, to_structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from pymatgen.core import Structure

    from materialsframework.tools.calculator import BaseCalculator

__authors__ = ["Doguhan Sariturk", "Elias P. Martin"]
__email__ = "dogu.sariturk@gmail.com"


class ElasticConstantsAnalyzer(BaseAnalyzer):
    """Calculate elastic constants and related properties from stress-strain data."""

    EQUIV = {
        "cubic": {
            (0, 0): ((1, 1), (2, 2)),
            (0, 1): ((0, 2), (1, 2)),
            (3, 3): ((4, 4), (5, 5)),
        },
        "hexagonal": {(0, 0): ((1, 1),), (0, 2): ((1, 2),), (3, 3): ((4, 4),)},
        "tetragonal": {(0, 0): ((1, 1),), (0, 2): ((1, 2),), (3, 3): ((4, 4),)},
        "trigonal": {(0, 0): ((1, 1),), (0, 2): ((1, 2),), (3, 3): ((4, 4),)},
    }

    def __init__(
        self,
        num_deform: int = 5,
        max_deform: float = 2,
        calculator: BaseCalculator | None = None,
        elastic_constant_transformation: ElasticConstantsDeformationTransformation | None = None,
    ) -> None:
        """Initializes the `ElasticConstantsAnalyzer` object.

        Args:
            num_deform (int, optional): The number of deformations to apply. Defaults to 5.
            max_deform (float, optional): The maximum deformation size in percent and degrees. Defaults to 2%.
            calculator (BaseCalculator | None, optional): The calculator used for structure relaxation and
                energy/stress calculations.
            elastic_constant_transformation (ElasticConstantsDeformationTransformation | None, optional): The
                transformation used to apply the symmetry-appropriate elastic deformations.
        """
        super().__init__(calculator)
        self.num_deform = num_deform
        self.max_deform = max_deform

        self._elastic_constant_transformation = elastic_constant_transformation

    @require_properties("energy", "stress", "forces")
    def calculate(
        self, structure: Structure | Atoms, is_relaxed: bool = False, relax_ions: bool = False, fmax_distort: float | None = None
    ) -> dict[str, float]:
        """Calculate elastic constants and derived properties.

        Args:
            structure (Structure | Atoms): The input structure to calculate the elastic constants of.
            is_relaxed (bool, optional): Whether to skip initial structure relaxation. Defaults to False.
            relax_ions (bool, optional): Whether to relax ions within each distorted fixed cell. Defaults to False.
            fmax_distort (float | None, optional): Force threshold in eV/Angstrom for distorted-cell relaxations.
                None uses the calculator's existing ``fmax``. Ignored when ``relax_ions`` is False. The original
                value is restored afterward. Defaults to None.

        Returns:
            dict[str, float]: Dictionary with keys:
                - ``C_ij`` entries: Elastic constants in GPa for all fitted tensor components.
                - ``youngs_modulus``: Young's modulus in GPa.
                - ``voigt_bulk_modulus``: Voigt bulk modulus in GPa.
                - ``voigt_shear_modulus``: Voigt shear modulus in GPa.
                - ``reuss_bulk_modulus``: Reuss bulk modulus in GPa.
                - ``reuss_shear_modulus``: Reuss shear modulus in GPa.
                - ``voigt_reuss_hill_bulk_modulus``: Voigt-Reuss-Hill bulk modulus in GPa.
                - ``voigt_reuss_hill_shear_modulus``: Voigt-Reuss-Hill shear modulus in GPa.
                - ``poisson_ratio``: Poisson ratio.
                - ``pugh_ratio``: Pugh ratio.
                - ``chen_vickers_hardness``: Chen-Vickers hardness in GPa.
                - ``debye_temperature``: Elastic estimate of the Debye temperature in K from the mean sound
                  velocity and atomic volume.
                - ``gruneisen_approx``: Dimensionless Belomestnykh–Tesleva estimate of the acoustic Grüneisen
                  parameter from the homogeneous Poisson ratio.
                - ``v_longitudinal``: Isotropic longitudinal sound velocity in m/s from the Voigt-Reuss-Hill bulk
                  and shear moduli and the structure's mass density.
                - ``v_transverse``: Isotropic transverse sound velocity in m/s from the Voigt-Reuss-Hill shear
                  modulus and the structure's mass density.
                - ``v_mean``: Debye mean sound velocity in m/s calculated from one longitudinal and two transverse
                  acoustic modes.

        Raises:
            ValueError: If required calculator properties are unavailable or fitted moduli are negative.
        """
        structure = self._ensure_relaxed(structure, is_relaxed)
        pmg_structure = to_structure(structure)
        structure = to_atoms(structure)

        prev_relax_cell = self.calculator.relax_cell
        prev_fmax = self.calculator.fmax
        self.calculator.relax_cell = False

        if fmax_distort is not None:
            self.calculator.fmax = fmax_distort

        try:
            structure.calc = self.calculator.calculator

            distorted_structures = self.elastic_constants_transformation.apply_transformation(structure)

            if relax_ions:
                distorted_structures = [to_atoms(self.calculator.relax(s)["final_structure"]) for s in distorted_structures]

            for distorted_structure in distorted_structures:
                distorted_structure.calc = self.calculator.calculator

            cij_order = elastic.get_cij_order(structure)
            cij, bij = elastic.get_elastic_tensor(
                cryst=structure,
                systems=distorted_structures,
            )
        finally:
            self.calculator.relax_cell = prev_relax_cell
            self.calculator.fmax = prev_fmax

        cij = np.asarray(cij, dtype=float) * EV_A3_TO_GPA

        elastic_tensor = self._build_elastic_tensor(cij, cij_order, structure)

        poisson_ratio = elastic_tensor.homogeneous_poisson
        pugh_ratio = elastic_tensor.g_vrh / elastic_tensor.k_vrh
        chen_vickers_hardness = 2.0 * (pugh_ratio**2 * elastic_tensor.g_vrh) ** 0.585 - 3.0
        debye_temperature = elastic_tensor.debye_temperature(pmg_structure)
        gruneisen_approx = 1.5 * (1.0 + poisson_ratio) / (2.0 - 3.0 * poisson_ratio)
        v_l = elastic_tensor.long_v(pmg_structure)
        v_t = elastic_tensor.trans_v(pmg_structure)
        v_mean = 3.0 ** (1.0 / 3.0) * (1.0 / v_l**3 + 2.0 / v_t**3) ** (-1.0 / 3.0)

        return {
            **dict(zip(cij_order, cij, strict=False)),
            "youngs_modulus": elastic_tensor.y_mod / 1e9,
            "voigt_bulk_modulus": elastic_tensor.k_voigt,
            "voigt_shear_modulus": elastic_tensor.g_voigt,
            "reuss_bulk_modulus": elastic_tensor.k_reuss,
            "reuss_shear_modulus": elastic_tensor.g_reuss,
            "voigt_reuss_hill_bulk_modulus": elastic_tensor.k_vrh,
            "voigt_reuss_hill_shear_modulus": elastic_tensor.g_vrh,
            "poisson_ratio": poisson_ratio,
            "pugh_ratio": pugh_ratio,
            "chen_vickers_hardness": chen_vickers_hardness,
            "debye_temperature": debye_temperature,
            "gruneisen_approx": gruneisen_approx,
            "v_longitudinal": v_l,
            "v_transverse": v_t,
            "v_mean": v_mean,
        }

    @lazy_property("_elastic_constant_transformation")
    def elastic_constants_transformation(
        self,
    ) -> ElasticConstantsDeformationTransformation:
        """Returns the transformation object used to apply distortions.

        Returns:
            ElasticConstantsDeformationTransformation: The transformation object used to apply distortions.
        """
        return ElasticConstantsDeformationTransformation(num_deform=self.num_deform, max_deform=self.max_deform)

    def _build_elastic_tensor(self, cij: Sequence[float], cij_order: Sequence[str], structure: Atoms) -> ElasticTensor:
        """Builds the elastic tensor from the given cij and cij_order.

        Args:
            cij (Sequence[float]): The list of elastic constants.
            cij_order (Sequence[str]): The order of the elastic constants.
            structure (Atoms): The input structure.

        Returns:
            ElasticTensor: The constructed elastic tensor.
        """
        elastic_tensor = np.zeros([6, 6])

        for val, sym in zip(cij, cij_order, strict=False):
            i, j = int(sym[2]) - 1, int(sym[3]) - 1
            elastic_tensor[i, j] = elastic_tensor[j, i] = val

        system = elastic.get_lattice_type(structure)[1].lower()
        for source, targets in self.EQUIV.get(system, {}).items():
            value = elastic_tensor[source]
            for p, q in targets:
                elastic_tensor[p, q] = elastic_tensor[q, p] = value

        # add the derived C66 if required
        if system in {"hexagonal", "trigonal"}:
            elastic_tensor[5, 5] = 0.5 * (elastic_tensor[0, 0] - elastic_tensor[0, 1])

        return ElasticTensor.from_voigt(elastic_tensor)
