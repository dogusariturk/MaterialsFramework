"""This module contains a class to calculate the elastic constant tensor of a given structure.

The `ElasticConstantsAnalyzer` class computes the elastic constant tensor of a structure using
energy-volume data and various deformation modes. The class also computes additional mechanical
properties such as bulk modulus, shear modulus, Poisson's ratio, and Pugh's ratio based on the
calculated elastic constants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymatgen.analysis.elasticity import ElasticTensor

from materialsframework.analysis.base import BaseAnalyzer
from materialsframework.analysis.utils import require_properties
from materialsframework.tools import elastic
from materialsframework.transformations.elastic_constants import (
    ElasticConstantsDeformationTransformation,
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


class ElasticConstantsAnalyzer(BaseAnalyzer):
    """A class used to calculate the elastic constant tensor for a given structure.

    Computes the elastic constant tensor from deformation and energy-volume data, along with derived
    mechanical properties such as bulk modulus, shear modulus, Young's modulus, and Poisson's ratio.
    """

    EQUIV = {
        "cubic": [
            ((0, 0), (1, 1), (2, 2)),
            ((0, 1), (0, 2), (1, 2)),
            ((3, 3), (4, 4), (5, 5)),
        ],
        "hexagonal": [((0, 0), (1, 1)), ((0, 2), (1, 2)), ((3, 3), (4, 4))],
        "tetragonal": [((0, 0), (1, 1)), ((0, 2), (1, 2)), ((3, 3), (4, 4))],
        "trigonal": [((0, 0), (1, 1)), ((0, 2), (1, 2)), ((3, 3), (4, 4))],
        "orthorhombic": [],
        "monoclinic": [],
        "triclinic": [],
    }

    SPECIAL = {"hexagonal", "trigonal"}  # need C66 = ½(C11–C12)

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
            calculator (BaseCalculator | None, optional): The calculator object used for energy calculations.
            elastic_constant_transformation (ElasticConstantsDeformationTransformation | None, optional): The
                transformation object used to apply cubic distortions.
        """
        super().__init__(calculator)
        self.num_deform = num_deform
        self.max_deform = max_deform

        self._elastic_constant_transformation = elastic_constant_transformation

    @require_properties("energy", "stress")
    def calculate(self, structure: Structure | Atoms, is_relaxed: bool = False) -> dict[str, float]:
        """Calculates the elastic constants of a given structure.

        Uses stress-strain data from a series of deformation modes to fit the elastic constants.

        Args:
            structure (Structure | Atoms): The input structure to calculate the elastic constants.
            is_relaxed (bool, optional): A flag to indicate whether the input structure is already relaxed. Defaults
                to False.

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

        Raises:
            ValueError: If the calculator object does not have the 'energy' and 'stress' properties implemented.
        """
        structure = self._ensure_relaxed(structure, is_relaxed)
        structure = structure.to_ase_atoms(msonable=False)

        prev_relax_cell = self.calculator.relax_cell
        self.calculator.relax_cell = False
        try:
            structure.calc = self.calculator.calculator

            distorted_structures = self.elastic_constants_transformation.apply_transformation(structure)

            for distorted_structure in distorted_structures:
                distorted_structure.calc = self.calculator.calculator

            cij_order = elastic.get_cij_order(structure)
            cij, bij = elastic.get_elastic_tensor(
                cryst=structure,
                systems=distorted_structures,
            )
        finally:
            self.calculator.relax_cell = prev_relax_cell

        cij = np.asarray(cij, dtype=float) * EV_A3_TO_GPA

        elastic_tensor = self._build_elastic_tensor(cij, cij_order, structure)

        return {
            **dict(zip(cij_order, cij, strict=False)),
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

        for block in self.EQUIV.get(sys := elastic.get_lattice_type(structure)[1].lower(), []):
            mean_val = np.mean([elastic_tensor[p, q] for p, q in block])
            for p, q in block:
                elastic_tensor[p, q] = elastic_tensor[q, p] = mean_val

        # add the derived C66 if required
        if sys in self.SPECIAL and elastic_tensor[5, 5] == 0:
            elastic_tensor[5, 5] = 0.5 * (elastic_tensor[0, 0] - elastic_tensor[0, 1])

        return ElasticTensor.from_voigt(elastic_tensor)
