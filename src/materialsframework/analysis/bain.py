"""
This module provides a class to perform a Bain transformation on a given structure.

The `BainPathAnalyzer` class calculates the potential energies along the Bain transformation path,
which describes the structural transition between body-centered cubic (BCC) and face-centered cubic (FCC)
phases. This transformation is essential for understanding phase stability and transformations in various
metallic systems.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from materialsframework.transformations.bain import BainDisplacementTransformation

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class BainPathAnalyzer:
    """
    A class used to analyze the Bain transformation path for a given structure.

    The `BainPathAnalyzer` class provides methods to perform a Bain transformation on an undeformed
    structure and calculate the potential energies at various c/a ratios along the transformation path.
    """

    def __init__(
        self,
        start: float = 0.89,
        stop: float = 1.4,
        step: float = 0.01,
        calculator: BaseCalculator | None = None,
        bain_transformation: BainDisplacementTransformation | None = None,
    ) -> None:
        """
        Initializes the `BainPathAnalyzer` object.

        Args:
            start (float, optional): The starting displacement value for the c/a ratio. Defaults to 0.89.
            stop (float, optional): The stopping displacement value for the c/a ratio. Defaults to 1.4.
            step (float, optional): The step size for incrementing the c/a ratio. Defaults to 0.01.
            calculator (BaseCalculator | None, optional): The calculator object used to compute potential energies.
                                                            Defaults to `M3GNetCalculator`.
            bain_transformation (BainDisplacementTransformation | None, optional): The transformation object used to
                                                                                      apply Bain displacements. If not provided,
                                                                                      a new instance is initialized.
        """
        self.start = start
        self.stop = stop
        self.step = step

        self.ase_adaptor = AseAtomsAdaptor()
        self._calculator = calculator
        self._bain_transformation = bain_transformation

    def calculate(
            self,
            structure: Structure | Atoms,
            is_relaxed: bool = False
    ) -> dict[str, list]:
        """
        Calculates the potential energies along the Bain Path for the given undeformed structure.

        This method applies the Bain transformation to the input structure, generating a series of deformed
        structures corresponding to different c/a ratios along the Bain path. It then calculates the potential
        energies of each deformed structure using the provided calculator.

        Args:
            structure (Structure | Atoms): The undeformed structure to be transformed and analyzed.
            is_relaxed (bool, optional): Whether the input structure is already relaxed. Defaults to False.

        Returns:
            dict[str, list]: A dictionary with the following keys:
                - "c_a_list": A list of c/a ratios corresponding to the deformed structures.
                - "energy_list": A list of potential energies for each deformed structure.

        Raises:
            ValueError: If the calculator object does not have the 'energy' property implemented.
        """
        if "energy" not in self.calculator.AVAILABLE_PROPERTIES:
            raise ValueError("The calculator object must have the 'energy' property implemented.")

        if isinstance(structure, Atoms):
            structure = self.ase_adaptor.get_structure(structure)

        if not is_relaxed:
            structure: Structure = self.calculator.relax(structure)["final_structure"]

        self.bain_transformation.apply_transformation(structure=structure)

        c_a_list, energy_list = zip(
                *[(c_a, self.calculator.calculate(structure=deformed_structure)["energy"])
                  for c_a, deformed_structure in self.bain_transformation.displaced_structures.items()])

        return {
                "c_a_list": c_a_list,
                "energy_list": energy_list
        }

    @property
    def calculator(self) -> BaseCalculator:
        """
        Returns the calculator instance used for energy calculations.

        If the calculator instance is not already initialized, this method creates a new `M3GNetCalculator` instance.

        Returns:
            BaseCalculator: The calculator object used for calculating energies.
        """
        if self._calculator is None:
            from materialsframework.calculators.m3gnet import M3GNetCalculator
            self._calculator = M3GNetCalculator()
        return self._calculator

    @property
    def bain_transformation(self) -> BainDisplacementTransformation:
        """
        Returns the Bain displacement transformation object used to apply Bain displacements.

        If the transformation instance is not already initialized, this method creates a new `BainDisplacementTransformation` instance.

        Returns:
            BainDisplacementTransformation: The transformation object used for Bain displacements.
        """
        if self._bain_transformation is None:
            self._bain_transformation = BainDisplacementTransformation(
                    start=self.start,
                    stop=self.stop,
                    step=self.step
            )
        return self._bain_transformation
