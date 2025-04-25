"""
This module contains a class to calculate the elastic constant tensor of a given structure.

The `ElasticConstantsAnalyzer` class computes the elastic constant tensor of a structure using
energy-volume data and various deformation modes. The class also computes additional mechanical
properties such as bulk modulus, shear modulus, Poisson's ratio, and Pugh's ratio based on the
calculated elastic constants.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from elastic import elastic
from pymatgen.analysis.elasticity import ElasticTensor

from materialsframework.transformations.elastic_constants import ElasticConstantsDeformationTransformation

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure
    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

eV_A3_to_GPa: float = 160.21766208


class ElasticConstantsAnalyzer:
    """
    A class used to calculate the elastic constant tensor for a given structure.

    The `ElasticConstantsAnalyzer` class provides methods to compute the elastic constant tensor
    of a structure using deformation and energy-volume data. In addition to the elastic constants,
    this class computes mechanical properties such as bulk modulus, shear modulus, Young's modulus,
    and Poisson's ratio.
    """

    EQUIV = {
            "cubic": [((0, 0), (1, 1), (2, 2)),
                      ((0, 1), (0, 2), (1, 2)),
                      ((3, 3), (4, 4), (5, 5))],
            "hexagonal": [((0, 0), (1, 1)),
                          ((0, 2), (1, 2)),
                          ((3, 3), (4, 4))],
            "tetragonal": [((0, 0), (1, 1)),
                           ((0, 2), (1, 2)),
                           ((3, 3), (4, 4))],
            "trigonal": [((0, 0), (1, 1)),
                         ((0, 2), (1, 2)),
                         ((3, 3), (4, 4))],
            "orthorhombic": [],
            "monoclinic": [],
            "triclinic": []
    }

    SPECIAL = {"hexagonal", "trigonal"}  # need C66 = ½(C11–C12)

    def __init__(
            self,
            num_deform: int = 5,
            max_deform: float = 2,
            fmax: float = 0.01,
            calculator: BaseCalculator | None = None,
            elastic_constant_transformation: ElasticConstantsDeformationTransformation | None = None
    ) -> None:
        """
        Initializes the `ElasticConstantsAnalyzer` object.

        Args:
            num_deform (int, optional): The number of deformations to apply. Defaults to 5.
            max_deform (float, optional): The maximum deformation size in percent and degrees. Defaults to 2%.
            fmax (float, optional): The maximum force for the calculator. Defaults to 0.01.
            calculator (BaseCalculator | None, optional): The calculator object used for energy calculations.
                                                          Defaults to `M3GNetCalculator`.
            elastic_constant_transformation (ElasticConstantsDeformationTransformation | None, optional): The transformation
                                                                                                    object used to apply
                                                                                                    cubic distortions.
        """
        self.num_deform = num_deform
        self.max_deform = max_deform
        self.fmax = fmax

        self._calculator = calculator
        self._elastic_constant_transformation = elastic_constant_transformation

    def calculate(
            self,
            structure: Structure,
            is_relaxed: bool = False
    ) -> dict[str, float]:
        """
        Calculates the elastic constants of a given structure.

        The method calculates the elastic constants of a structure using stress-strain data and various
        deformation modes. The resulting elastic constants are returned as a dictionary with the elastic
        constant names as keys and the corresponding values in GPa.

        Args:
            structure (Structure): The input structure to calculate the elastic constants.
            is_relaxed (bool, optional): A flag to indicate whether the input structure is already relaxed.
                                         Defaults to False.

        Returns:
            dict[str, float]: A dictionary containing the elastic constants and their values in GPa.

        Raises:
            ValueError: If the calculator object does not have the 'energy' property implemented.
        """
        if "energy" not in self.calculator.AVAILABLE_PROPERTIES:
            raise ValueError("The calculator object must have the 'energy' property implemented.")

        if not is_relaxed:
            structure = self.calculator.relax(structure)["final_structure"]

        structure = structure.to_ase_atoms(msonable=False)
        self.calculator.relax_cell = False
        structure.calc = self.calculator.calculator

        self.elastic_constants_transformation.apply_transformation(structure)

        cij_order = elastic.get_cij_order(structure)
        Cij, Bij = elastic.get_elastic_tensor(
                cryst=structure,
                systems=self.elastic_constants_transformation.distorted_structures
        )
        Cij *= eV_A3_to_GPa

        elastic_tensor = self._build_elastic_tensor(Cij, cij_order, structure)

        return {
            **{i: j for i, j in zip(cij_order, Cij)},
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
            from materialsframework.calculators.m3gnet import M3GNetCalculator
            self._calculator = M3GNetCalculator(fmax=self.fmax)
        return self._calculator

    @property
    def elastic_constants_transformation(self) -> ElasticConstantsDeformationTransformation:
        """
        Returns the transformation object used to apply distortions.

        If the transformation object is not already initialized, this method creates a new `ElasticConstantsDeformationTransformation` instance.

        Returns:
            ElasticConstantsDeformationTransformation: The transformation object used to apply distortions.
        """
        if self._elastic_constant_transformation is None:
            self._elastic_constant_transformation = ElasticConstantsDeformationTransformation(
                num_deform=self.num_deform,
                max_deform=self.max_deform
            )
        return self._elastic_constant_transformation

    def _build_elastic_tensor(
            self,
            Cij: list,
            cij_order: list,
            structure: Atoms
    ) -> ElasticTensor:
        """
        Builds the elastic tensor from the given Cij and cij_order.

        Args:
            Cij (list): The list of elastic constants.
            cij_order (list): The order of the elastic constants.
            structure (Atoms): The input structure.

        Returns:
            ElasticTensor: The constructed elastic tensor.
        """
        elastic_tensor = np.zeros([6, 6])

        for val, sym in zip(Cij, cij_order):
            i, j = int(sym[2]) - 1, int(sym[3]) - 1
            elastic_tensor[i, j] = elastic_tensor[j, i] = val

        for block in self.EQUIV.get(sys := elastic.get_lattice_type(structure)[1], []):
            ref = elastic_tensor[block[0]]
            for (p, q) in block:
                elastic_tensor[p, q] = elastic_tensor[q, p] = ref

        # add the derived C66 if required
        if sys in self.SPECIAL and elastic_tensor[5, 5] == 0:
            elastic_tensor[5, 5] = 0.5 * (elastic_tensor[0, 0] - elastic_tensor[0, 1])

        return ElasticTensor.from_voigt(elastic_tensor)
