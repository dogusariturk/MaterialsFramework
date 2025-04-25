"""
This module provides a class for performing calculations using the MACE potential.

The `MACECalculator` class is designed to calculate properties such as potential energy, free energy,
forces, and stresses, and to perform structure relaxation using a specified MACE model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from pathlib import Path

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class MACECalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the MACE potential.

    The `MACECalculator` class supports the calculation of properties such as potential energy,
    free energy, forces, and stresses. It also allows for the relaxation of structures using
    a specified MACE model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "free_energy", "forces", and "stress".
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "node_energy", "forces", "stress"]

    def __init__(
            self,
            model: str | Path = "medium-mpa-0",
            include_dipoles: bool = False,
            device: Literal["cuda", "cpu", "mps"] = "cpu",
            default_dtype: str = "",
            model_type: Literal["MACE", "DipoleMACE", "EnergyDipoleMACE"] = "MACE",
            **kwargs
    ) -> None:
        """
        Initializes the MACECalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined MACE model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters for
        the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model (str | Path, optional): The MACE model to use. This can be the name of a predefined model
                (e.g., "medium-omat-0"), a path to a custom model file, or a URL. Defaults to "medium-mpa-0".
            include_dipoles (bool, optional): Determines whether dipole properties are included in the model. Defaults to False.
            device (Literal["cuda", "cpu", "mps"], optional): The device to use for calculations. Defaults to "cpu".
            default_dtype (str, optional): The default data type to be used for the model. Defaults to an empty string,
                meaning the default data type of the model will be used.
            model_type (Literal["MACE", "DipoleMACE", "EnergyDipoleMACE"], optional): The type of MACE model to use. Defaults to "MACE".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Examples:
            >>> mace_calculator = MACECalculator(model="large", device="cuda")

        Note:
            The remaining parameters for the MACE potential are set to their default values, which are appropriate for general use cases.
            If needed, they can be adjusted based on specific calculation requirements.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # MACE specific attributes
        self.model = model
        self.device = device
        self.default_dtype = default_dtype
        self.model_type = model_type

        if include_dipoles:
            self.__class__.AVAILABLE_PROPERTIES.append("dipole")

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """
        Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the specified MACE model and other
        relevant attributes: `device`, `default_dtype`, and `model_type`. If the Calculator object
        has already been created, it will return the existing instance.

        Returns:
            Calculator: The ASE Calculator object configured with the MACE potential.
        """
        if self._calculator is None:
            from mace.calculators import mace_mp
            self._calculator = mace_mp(
                    model=self.model,
                    device=self.device,
                    default_dtype=self.default_dtype,
                    model_type=self.model_type,
            )
        return self._calculator
