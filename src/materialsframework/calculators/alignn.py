"""
This module provides a class for performing calculations and structure relaxation using the ALIGNN-FF potential.

The `AlignnCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and to perform structure relaxation using a specified ALIGNN-FF model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class AlignnCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the ALIGNN-FF potential.

    The `AlignnCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified ALIGNN-FF model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stresses".
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
            self,
            path: str | None = None,
            model_filename="best_model.pt",
            config_filename="config.json",
            device: Literal["cuda", "cpu", "mps"] = "cpu",
            **kwargs
    ) -> None:
        """
        Initializes the AlignnCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined ALIGNN-FF model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            path (str | None): The path to the directory containing the ALIGNN-FF model files.
                               If None, 'v12.2.2024_dft_3d_307k' model will be used.
            model_filename (str): The filename of the model file. Defaults to "best_model.pt".
            config_filename (str): The filename of the configuration file. Defaults to "config.json".
            device (Literal["cuda", "cpu", "mps"], optional): The device to use for calculations. Defaults to "cpu".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # ALIGNN-FF specific attributes
        self.path = path
        self.model_filename = model_filename
        self.config_filename = config_filename
        self.device = device

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """
        Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the ALIGNN-FF potential and other settings
        specified during the initialization of this calculator. The Calculator object is then returned
        to the caller. If the Calculator object has already been created, it is returned directly.

        Returns:
            Calculator: The ASE Calculator object configured with the ALIGNN-FF potential.
        """
        if self._calculator is None:
            from alignn.ff.calculators import AlignnAtomwiseCalculator
            self._calculator = AlignnAtomwiseCalculator(
                path=self.path,
                model_filename=self.model_filename,
                config_filename=self.config_filename,
                device=self.device,
            )
        return self._calculator
