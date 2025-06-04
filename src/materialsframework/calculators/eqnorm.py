"""
This module provides a class for performing calculations and structure relaxation using the Eqnorm potential.

The `EqnormCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and to perform structure relaxation using a specified Eqnorm model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class EqnormCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the Eqnorm potential.

    The `EqnormCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified Eqnorm model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stresses".

    References:

    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model_variant: Literal["eqnorm-mptrj", "eqnorm-pro-mptrj"] = "eqnorm-mptrj",
        model_name: str = "eqnorm",
        train_progress: str = "1.0",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """
        Initializes the EqnormCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined Eqnorm model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model_name (str): The name of the Eqnorm model to use for calculations.
            model_variant (str): The variant of the Eqnorm model to use.
            train_progress (str, optional): The training progress version of the model. Defaults to "1.0".
            device (Literal["cuda", "cpu"], optional): The device to use for calculations. Defaults to "cpu".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # Eqnorm specific attributes
        self.model_name = model_name
        self.model_variant = model_variant
        self.train_progress = train_progress
        self.device = device

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """
        Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the Eqnorm potential and other settings
        specified during the initialization of this calculator. The Calculator object is then returned
        to the caller. If the Calculator object has already been created, it is returned directly.

        Returns:
            Calculator: The ASE Calculator object configured with the AlphaNet potential.
        """
        if self._calculator is None:
            from eqnorm.calculator import EqnormCalculator
            self._calculator = EqnormCalculator(
                    model_name=self.model_name,
                    model_variant=self.model_variant,
                    train_progress=self.train_progress,
                    device=self.device
            )
        return self._calculator
