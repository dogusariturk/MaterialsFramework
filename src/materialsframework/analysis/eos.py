"""
This module provides a class to perform an Equation of State (EOS) analysis on a given structure.

The `EOSAnalyzer` class allows users to perform an EOS analysis by applying a series of volume changes
to a structure and calculating the corresponding energies. The resulting data is used to fit a chosen
equation of state (EOS), providing insights into the mechanical properties of the material, such as the
bulk modulus.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ase import Atoms
from pymatgen.analysis.eos import EOS
from pymatgen.io.ase import AseAtomsAdaptor

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
            start: float = -0.1,
            stop: float = 0.1,
            num: int = 11,
            eos_name: str = "birch_murnaghan",
            calculator: BaseCalculator | None = None,
            eos_transformation: EOSTransformation | None = None
    ) -> None:
        """
        Initializes the `EOSAnalyzer` object.

        Args:
            start (float, optional): The starting strain value to apply to the structure. Defaults to -0.1.
            stop (float, optional): The stopping strain value to apply to the structure. Defaults to 0.1.
            num (int, optional): The number of strain values to generate between the start and stop. Defaults to 11.
            eos_name (str, optional): The name of the equation of state (EOS) used for fitting. Defaults to "birch_murnaghan".
            calculator (BaseCalculator | None, optional): The calculator used for energy calculations. Defaults to `M3GNetCalculator`.
            eos_transformation (EOSTransformation | None, optional): The transformation used to generate deformed structures.
                                                                         Defaults to `EOSTransformation`.
        """
        self.start = start
        self.stop = stop
        self.num = num
        self.eos_name = eos_name

        self.ase_adaptor = AseAtomsAdaptor()
        self._calculator = calculator
        self._eos_transformation = eos_transformation

    def calculate(
            self,
            structure: Structure | Atoms,
            is_relaxed: bool = False
    ) -> dict[str, list | float]:
        """
        Calculates the potential energies and volumes to construct the EOS for the given undeformed structure.

        This method applies a series of volume deformations to the input structure, generating a set of strained
        structures. It then calculates the corresponding potential energies and fits the data to the specified
        equation of state (EOS).

        Args:
            structure (Structure | Atoms): The undeformed structure to be analyzed.
            is_relaxed (bool, optional): Whether the structure is already relaxed. Defaults to False.

        Returns:
            dict[str, list | float]: A dictionary with the following keys:
                - "strains": A list of strain values corresponding to the deformed structures.
                - "volumes": A list of volumes for each deformed structure.
                - "energies": A list of potential energies for each deformed structure.
                - "e0": The minimum energy of the system.
                - "b0": The bulk modulus in units of energy/unit of volume^3.
                - "b0_GPa": The bulk modulus in GPa.
                - "b1": The derivative of bulk modulus with respect to pressure.
                - "v0": The minimum volume of the system in Ang^3.

        Raises:
            ValueError: If the calculator object does not have the 'energy' property implemented.
        """
        if "energy" not in self.calculator.AVAILABLE_PROPERTIES:
            raise ValueError("The calculator object must have the 'energy' property implemented.")

        if isinstance(structure, Atoms):
            structure = self.ase_adaptor.get_structure(structure)

        if not is_relaxed:
            self.calculator.relax_cell = True
            structure: Structure = self.calculator.relax(structure)["final_structure"]
            self.calculator.relax_cell = False

        self.calculator.relax_cell = False
        self.eos_transformation.apply_transformation(structure)

        strain_list, volume_list, energy_list = zip(
                *[(strain, deformed_structure.volume, self.calculator.relax(structure=deformed_structure)["energy"]) for
                  strain, deformed_structure in self.eos_transformation.structures.items()])

        eos = EOS(eos_name=self.eos_name)
        eos_fit = eos.fit(volumes=volume_list, energies=energy_list)

        return {
                "strains": strain_list,
                "volumes": volume_list,
                "energies": energy_list,
                "e0": eos_fit.e0,
                "b0": eos_fit.b0,
                "b0_GPa": eos_fit.b0_GPa,
                "b1": eos_fit.b1,
                "v0": eos_fit.v0,
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
            self._eos_transformation = EOSTransformation(
                    start=self.start,
                    stop=self.stop,
                    num=self.num
            )
        return self._eos_transformation
