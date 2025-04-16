"""
This module contains a class to calculate the elastic constant tensor of a given structure.

The `ElasticConstantsAnalyzer` class computes the elastic constant tensor of a structure using
energy-volume data and various deformation modes. The class also computes additional mechanical
properties such as bulk modulus, shear modulus, Poisson's ratio, and Pugh's ratio based on the
calculated elastic constants.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from elastic import elastic

from materialsframework.calculators.m3gnet import M3GNetCalculator
from materialsframework.transformations.elastic_constants import ElasticConstantsDeformationTransformation

if TYPE_CHECKING:
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

    def __init__(
            self,
            calculator: BaseCalculator | None = None,
            elastic_constant_transformation: ElasticConstantsDeformationTransformation | None = None
    ) -> None:
        """
        Initializes the `ElasticConstantsAnalyzer` object.

        Args:
            calculator (BaseCalculator | None, optional): The calculator object used for energy calculations.
                                                          Defaults to `M3GNetCalculator`.
            elastic_constant_transformation (ElasticConstantsDeformationTransformation | None, optional): The transformation
                                                                                                    object used to apply
                                                                                                    cubic distortions.
        """
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

        return {
                i: j * eV_A3_to_GPa for i, j in zip(cij_order, Cij)
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
            self._calculator = M3GNetCalculator(fmax=0.01)
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
            self._elastic_constant_transformation = ElasticConstantsDeformationTransformation()
        return self._elastic_constant_transformation
