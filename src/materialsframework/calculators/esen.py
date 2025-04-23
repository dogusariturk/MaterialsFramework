"""
This module provides a class for performing calculations and structure relaxation using the eSEN potential.

The `eSENCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and to perform structure relaxation using a specified eSEN model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class eSENCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the eSEN potential.

    The `eSENCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified eSEN model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stress".
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
            self,
            model_name: str = 'eSEN-30M-OAM',
            checkpoint_path: str | None = None,
            local_cache: str = "~/.cache/esen/",
            device: Literal["cpu", "cuda"] = "cpu",
            seed: int | None = None,
            **kwargs
    ) -> None:
        """
        Initializes the eSENCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined eSEN model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model_name (str): The name of the eSEN model to use for calculations. Defaults to 'eSEN-30M-OAM'.
            checkpoint_path (str, optional): The path to the checkpoint file for the eSEN model.
            local_cache (str): The path to the local cache directory for the eSEN model. Defaults to "~/.cache/esen/".
            device (Literal["cpu", "cuda"], optional): The device to use for the calculations. Defaults to "cpu".
            seed (int, optional): The seed value for the model.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # eSEN specific attributes
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.local_cache = local_cache
        self.device = device
        self.seed = seed

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """
        Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the EqV2 potential and other settings
        specified during the initialization of this calculator. The Calculator object is then returned
        to the caller. If the Calculator object has already been created, it is returned directly.

        Returns:
            Calculator: The ASE Calculator object configured with the EqV2 potential.
        """
        if self._calculator is None:
            from fairchem.core import OCPCalculator
            self._calculator = OCPCalculator(
                    model_name=self.model_name,
                    checkpoint_path=self.checkpoint_path,
                    local_cache=self.local_cache,
                    cpu=self.device != "cuda",
                    seed=self.seed,
            )
        return self._calculator
