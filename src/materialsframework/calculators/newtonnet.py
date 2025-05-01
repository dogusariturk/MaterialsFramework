"""
This module provides a class for performing calculations and structure relaxation using the NewtonNet potential.

The `NewtonNetCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and to perform structure relaxation using a specified NewtonNet model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class NewtonNetCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the NewtonNet potential.

    The `NewtonNetCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified NewtonNet model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", "hessian", and "stresses".

    References:
        - NewtonNet: https://doi.org/10.1039/D2DD00008C
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "hessian", "stress"]

    def __init__(
            self,
            model_path: str | Literal["ani1", "ani1x", "t1x"] = "t1x",
            properties: list = ["energy", "free_energy", "forces", "hessian", "stress"],
            device: Literal["cpu", "cuda"] = "cpu",
            precision: Literal["float64", "float32", "float16"] = "float32",
            **kwargs
    ) -> None:
        """
        Initializes the NewtonNetCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined NewtonNet model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model_path (str | Literal["ani1", "ani1x", "t1x"]): Path to the NewtonNet model or a predefined model name. Defaults to "t1x"
            properties (list): List of properties to calculate, such as "energy", "forces", etc.
            device (Literal["cpu", "cuda"]): The device to use for calculations. Defaults to "cpu".
            precision (Literal["float64", "float32", "float16"]): Floating-point precision of the calculations. Defaults to "float32".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # NewtonNet specific attributes
        self.model_path = model_path
        self.properties = properties
        self.device = device
        self.precision = precision

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """
        Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the NewtonNet potential and other settings
        specified during the initialization of this calculator. The Calculator object is then returned
        to the caller. If the Calculator object has already been created, it is returned directly.

        Returns:
            Calculator: The ASE Calculator object configured with the NewtonNet potential.
        """
        if self._calculator is None:
            from newtonnet.utils.ase_interface import MLAseCalculator as NewtonNetASECalculator
            self._calculator = NewtonNetASECalculator(
                    model_path=self.model_path,
                    properties=self.properties,
                    device=self.device,
                    precision=self.precision
            )
        return self._calculator
