"""
This module provides a class to perform an Equation of State (EOS) analysis on a given structure.
"""
from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

from pymatgen.analysis.eos import EOS

from materialsframework.calculators.m3gnet import M3GNetCalculator, M3GNetRelaxer
from materialsframework.transformations.eos import EOSTransformation

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.tools.typing import Calculator, Relaxer

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class EOSAnalyzer:
    """
    A class used to represent an EOSAnalyzer.

    This class provides methods to perform an Equation of State (EOS) analysis on a given structure.
    """

    def __init__(
            self,
            eos_name: str = "birch_murnaghan",
            relaxer: Optional[Relaxer] = None,
            calculator: Optional[str] = None,
            eos_transformation: Optional[EOSTransformation] = None
    ) -> None:
        """
        Initializes the EOSAnalyzer.

        Args:
            eos_name (str): The name of the EOS to fit. Default is "birch_murnaghan".
            relaxer (Optional[Relaxer]): The Relaxer object to use for relaxation. Default is M3GNetRelaxer.
            calculator (Optional[str]): The Calculator object to use for calculations. Default is M3GNetCalculator
            eos_transformation (Optional[EOSTransformation]): The EOS transformation object. Default is EOSTransformation.
        """
        self._eos_name = eos_name
        self._relaxer = relaxer
        self._calculator = calculator

        self._eos_transformation = eos_transformation

    def calculate(self, undeformed_structure: Structure, is_relaxed: bool = False) -> dict:
        """
        Calculates the potential energies to construct the EOS for the given undeformed structure.

        Args:
            undeformed_structure (Structure): The undeformed structure.
            is_relaxed (bool): Whether the structure is already relaxed. Defaults to False.

        Returns:
            dict: A dictionary containing the strains, volumes, energies, and EOS fitting results.
        """
        undeformed_structure = self.relaxer.relax(structure=undeformed_structure)["final_structure"] if not is_relaxed else undeformed_structure

        self.eos_transformation.apply_transformation(undeformed_structure)

        strain_list, volume_list, energy_list = zip(
                *[(strain, deformed_structure.volume, self.calculator.calculate(structure=deformed_structure)["potential_energy"]) for
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
    def relaxer(self) -> Relaxer:
        """
        Returns the Relaxer instance.

        If the relaxer instance is not already created, it creates a new M3GNetRelaxer instance
        and returns it. Otherwise, it returns the existing relaxer instance.

        Returns:
            Relaxer: The Relaxer instance.
        """
        if self._relaxer is None:
            self._relaxer = M3GNetRelaxer()
        return self._relaxer

    @property
    def calculator(self) -> Calculator:
        """
        Returns the Calculator instance.

        If the calculator instance is not already created, it creates a new M3GNetCalculator
        instance with the specified potential and returns it. Otherwise, it returns the existing
        calculator instance.

        Returns:
            Calculator: The Calculator instance.
        """
        if self._calculator is None:
            self._calculator = M3GNetCalculator()
        return self._calculator

    @property
    def eos_transformation(self) -> EOSTransformation:
        """
        Gets the EOS transformation object.

        Returns:
            EOSTransformation: The EOS transformation object.
        """
        if self._eos_transformation is None:
            self._eos_transformation = EOSTransformation()
        return self._eos_transformation
