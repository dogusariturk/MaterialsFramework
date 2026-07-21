"""This module provides a class to perform an Equation of State (EOS) analysis on a given structure.

The `EOSAnalyzer` class applies a series of volume changes to a structure, calculates the corresponding
energies, and fits the resulting data to a chosen equation of state (EOS) to obtain mechanical properties
such as the bulk modulus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pymatgen.analysis.eos import EOS

from materialsframework.analysis.base import BaseAnalyzer
from materialsframework.analysis.utils import require_properties
from materialsframework.transformations.eos import EOSTransformation
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class EOSAnalyzer(BaseAnalyzer):
    """A class used to perform Equation of State (EOS) analysis for a given structure.

    Generates a series of structures with varying volumes via deformation transformations, calculates
    their energies, and fits the energy-volume data to an EOS (such as Birch-Murnaghan) to determine
    properties such as the bulk modulus.
    """

    def __init__(
        self,
        start: float = -0.1,
        stop: float = 0.1,
        num: int = 11,
        eos_name: Literal[
            "murnaghan",
            "birch",
            "birch_murnaghan",
            "pourier_tarantola",
            "vinet",
            "deltafactor",
            "numerical_eos"
        ] = "birch_murnaghan",
        calculator: BaseCalculator | None = None,
        eos_transformation: EOSTransformation | None = None,
    ) -> None:
        """Initializes the `EOSAnalyzer` object.

        Args:
            start (float, optional): The starting strain value to apply to the structure. Defaults to -0.1.
            stop (float, optional): The stopping strain value to apply to the structure. Defaults to 0.1.
            num (int, optional): The number of strain values to generate between the start and stop. Defaults to 11.
            eos_name (str, optional): The name of the equation of state (EOS) used for fitting. Defaults to "birch_murnaghan".
            calculator (BaseCalculator | None, optional): The calculator used for energy calculations.
            eos_transformation (EOSTransformation | None, optional): The transformation used to generate deformed
                structures. Defaults to `EOSTransformation`.
        """
        super().__init__(calculator)
        self.start = start
        self.stop = stop
        self.num = num
        self.eos_name = eos_name

        self._eos_transformation = eos_transformation

    @require_properties("energy")
    def calculate(self, structure: Structure | Atoms, is_relaxed: bool = False) -> dict[str, list | float]:
        """Calculates the potential energies and volumes to construct the EOS for the given undeformed structure.

        Applies a series of volume deformations to the input structure, calculates the potential energy
        of each strained structure, and fits the data to the specified equation of state (EOS).

        Args:
            structure (Structure | Atoms): The undeformed structure to be analyzed.
            is_relaxed (bool, optional): Whether the structure is already relaxed. Defaults to False.

        Returns:
            dict[str, list | float]: Dictionary with keys:
                - ``volumes``: Volume for each deformed structure.
                - ``energies``: Potential energy for each deformed structure.
                - ``e0``: Minimum energy from EOS fit.
                - ``b0``: Bulk modulus in eV/Å³.
                - ``b0_GPa``: Bulk modulus in GPa.
                - ``b1``: Pressure derivative of bulk modulus.
                - ``v0``: Equilibrium volume in Å³.

        Raises:
            ValueError: If the calculator object does not have the 'energy' property implemented.
        """
        structure = self._ensure_relaxed(structure, is_relaxed)

        structures = self.eos_transformation.apply_transformation(structure)

        prev_relax_cell = self.calculator.relax_cell
        self.calculator.relax_cell = False
        try:
            volume_list, energy_list = map(
                list,
                zip(
                    *[
                        (
                            result["final_structure"].volume,
                            result["energy"],
                        )
                        for deformed_structure in structures
                        for result in [self.calculator.relax(structure=deformed_structure)]
                    ],
                    strict=False,
                ),
            )
        finally:
            self.calculator.relax_cell = prev_relax_cell

        eos = EOS(eos_name=self.eos_name)
        eos_fit = eos.fit(volumes=volume_list, energies=energy_list)

        return {
            "volumes": volume_list,
            "energies": energy_list,
            "e0": eos_fit.e0,
            "b0": eos_fit.b0,
            "b0_GPa": eos_fit.b0_GPa,
            "b1": eos_fit.b1,
            "v0": eos_fit.v0,
        }

    @lazy_property("_eos_transformation")
    def eos_transformation(self) -> EOSTransformation:
        """Returns the EOS transformation object used to generate deformed structures.

        Returns:
            EOSTransformation: The transformation object used for EOS analysis.
        """
        return EOSTransformation(start=self.start, stop=self.stop, num=self.num)
