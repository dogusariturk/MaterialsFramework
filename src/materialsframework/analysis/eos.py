"""
This module provides a class to perform an Equation of State (EOS) analysis on a given structure.

The `EOSAnalyzer` class allows users to perform an EOS analysis by applying a series of volume changes
to a structure and calculating the corresponding energies. The resulting data is used to fit a chosen
equation of state (EOS), providing insights into the mechanical properties of the material, such as the
bulk modulus.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pymatgen.analysis.eos import EOS

from materialsframework.calculators.m3gnet import M3GNetCalculator
from materialsframework.transformations.eos import EOSTransformation

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class EOSAnalyzer:
    """
    A class used to perform Equation of State (EOS) analysis for a given structure.

    The `EOSAnalyzer` class provides methods to fit energy-volume data to an EOS (such as Birch-Murnaghan)
    for determining material properties like the bulk modulus. The class uses deformation transformations
    to create a series of structures with varying volumes and calculates their corresponding energies.
    """

    def __init__(
            self,
            eos_name: str = "birch_murnaghan",
            calculator: BaseCalculator | None = None,
            eos_transformation: EOSTransformation | None = None
    ) -> None:
        """
        Initializes the `EOSAnalyzer` object.

        Args:
            eos_name (str, optional): The name of the equation of state (EOS) used for fitting. Defaults to "birch_murnaghan".
            calculator (BaseCalculator | None, optional): The calculator used for energy calculations. Defaults to `M3GNetCalculator`.
            eos_transformation (EOSTransformation | None, optional): The transformation used to generate deformed structures.
                                                                         Defaults to `EOSTransformation`.
        """
        self._eos_name = eos_name
        self._calculator = calculator

        self._eos_transformation = eos_transformation

    def calculate(
            self,
            undeformed_structure: Structure,
            is_relaxed: bool = False
    ) -> dict[str, list]:
        """
        Calculates the potential energies and volumes to construct the EOS for the given undeformed structure.

        This method applies a series of volume deformations to the input structure, generating a set of strained
        structures. It then calculates the corresponding potential energies and fits the data to the specified
        equation of state (EOS).

        Args:
            undeformed_structure (Structure): The undeformed structure to be analyzed.
            is_relaxed (bool, optional): Whether the structure is already relaxed. Defaults to False.

        Returns:
            dict[str, list]: A dictionary containing the calculated strains, volumes, energies, and the EOS fitting results.
        """
        if "energy" not in self.calculator.AVAILABLE_PROPERTIES:
            raise ValueError("The calculator object must have the 'energy' property implemented.")

        undeformed_structure = self.calculator.relax(structure=undeformed_structure)["final_structure"] if not is_relaxed else undeformed_structure

        self.eos_transformation.apply_transformation(undeformed_structure)

        strain_list, volume_list, energy_list = zip(
                *[(strain, deformed_structure.volume, self.calculator.calculate(structure=deformed_structure)["energy"]) for
                  strain, deformed_structure in self.eos_transformation.structures.items()])

        eos = EOS(eos_name=self._eos_name)
        eos_fit = eos.fit(volumes=volume_list, energies=energy_list)

        return {
                "strains": strain_list,
                "volumes": volume_list,
                "energies": energy_list,
                "eos_results": eos_fit.results
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
    def eos_transformation(self) -> EOSTransformation:
        """
        Returns the EOS transformation object used to generate deformed structures.

        If the transformation instance is not already initialized, this method creates a new `EOSTransformation` instance.

        Returns:
            EOSTransformation: The transformation object used for EOS analysis.
        """
        if self._eos_transformation is None:
            self._eos_transformation = EOSTransformation()
        return self._eos_transformation
