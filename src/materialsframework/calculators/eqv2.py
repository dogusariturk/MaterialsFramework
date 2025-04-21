"""
This module provides a class for performing calculations and structure relaxation using the EqV2 potential.

The `EqV2Calculator` class is designed to calculate properties such as potential energy, forces,
stresses, and to perform structure relaxation using a specified EqV2 model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class EqV2Calculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the EqV2 potential.

    The `EqV2Calculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified EqV2 model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stress".
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
            self,
            checkpoint_path: str | None = None,
            model_name: str | None = None,
            local_cache: str | None = None,
            device: Literal["cpu", "cuda"] = "cpu",
            seed: int | None = 42,
            **kwargs
    ) -> None:
        """
        Initializes the EqV2Calculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined EqV2 model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            checkpoint_path (str, optional): The path to the checkpoint file for the EqV2 model.
            model_name (str, optional): The name of the EqV2 model to use.
            local_cache (str, optional): The path to the local cache directory for the EqV2 model.
            device (Literal["cpu", "cuda"], optional): The device to use for the calculations. Defaults to "cpu".
            seed (int, optional): The seed value for the model. Defaults to 42.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # EqV2 specific attributes
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
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
                    checkpoint_path=self.checkpoint_path,
                    model_name=self.model_name,
                    local_cache=self.local_cache,
                    cpu=self.device != "cuda",
                    seed=self.seed,
            )
        return self._calculator
