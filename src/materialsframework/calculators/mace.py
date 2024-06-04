"""
This module provides classes to perform calculations using the MACE potential.
"""
from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING, Union

from mace.calculators import MACECalculator as ASEMACECalculator

from materialsframework.calculators.typing import Calculator

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class MACECalculator(Calculator):
    """
    A class used to represent a MACE Calculator.

    This class provides methods to perform calculations using the MACE potential.
    """

    def __init__(
            self,
            device: Optional[str] = None,
            default_dtype: Optional[str] = None,
            model: str = "2023-12-03-mace-128-L1_epoch-199.model") -> None:
        """
        Initialize a MACECalculator instance.

        Args:
            device: The device to use for calculations (cuda or cpu).
            default_dtype: The default data type to use for calculations (float64 or float32).
            model: The model to use for calculations. Defaults to "2023-12-03-mace-128-L1_epoch-199.model".

        Examples:
            >>> calculator = MACECalculator(device="cpu", default_dtype="float32")
            >>> calculator = MACECalculator(device="cuda", default_dtype="float64", model="2023-12-03-mace-128-L1_epoch-199.model")

        Note:
            The remaining values for the arguments are set to the default values for the MACE potential.
        """
        self._device = device
        self._default_dtype = default_dtype
        self._model: str = model

        self._calculator = None
        self._potential = None

    @property
    def potential(self) -> str:
        """
        Returns the MACE potential path associated with this instance.

        If the potential has not been initialized yet, it will be loaded
        using the model attribute of this instance.

        Returns:
            str: The path to MACE potential associated with this instance.
        """
        if self._potential is None:
            models = {
                    "small": "2023-12-10-mace-128-L0_energy_epoch-249.model",
                    "medium": "2023-12-03-mace-128-L1_epoch-199.model",
                    "large": "MACE_MPtrj_2022.9.model",
            }
            model_file = models.get(self._model, self._model)
            self._potential = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/", model_file))
        return self._potential

    @property
    def calculator(self) -> ASEMACECalculator:
        """
        Returns the ASE MACECalculator associated with this instance.

        If the calculator has not been initialized yet, it will be created
        using the potential attribute of this instance.

        Returns:
            ASEMACECalculator: The ASE MACECalculator associated with this instance.
        """
        if self._calculator is None:
            self._calculator = ASEMACECalculator(
                    model_paths=self.potential,
                    device=self._device,
                    default_dtype=self._default_dtype,
            )
        return self._calculator

    def calculate(
            self,
            structure: Structure
    ) -> dict[str, Union[float, ArrayLike]]:
        """
        Calculate the potential energy, free energy, forces, and stresses
        of a structure using the MACE potential.

        Args:
            structure: The input structure.

        Returns:
            dict[str, Union[float, ArrayLike]]: A dictionary containing the calculated properties.

        Examples:
            >>> calculator = MACECalculator(device="cuda", default_dtype="float64")
            >>> struct = Structure.from_file("POSCAR")
            >>> calculation_results = calculator.calculate(structure=struct)
        """
        atoms = structure.to_ase_atoms()
        self.calculator.calculate(atoms=atoms)

        return {
                "potential_energy": self.calculator.results["energy"],
                "free_energy": self.calculator.results["free_energy"],
                "forces": self.calculator.results["forces"],
                "stress": self.calculator.results["stress"],
        }
