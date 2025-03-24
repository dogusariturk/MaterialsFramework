"""
This module provides a class for performing calculations and structure relaxation using the AlphaNet potential.

The `AlphaNetCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and magnetic moments, and to perform structure relaxation using a specified AlphaNet model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import torch

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class AlphaNetCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the AlphaNet potential.

    The `AlphaNetCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified AlphaNet model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stresses".
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
            self,
            config: str,
            checkpoint: str,
            device: Literal["cuda", "cpu", "mps"] = "cpu",
            **kwargs
    ) -> None:
        """
        Initializes the AlphaNetCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined AlphaNet model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            config (str): The path to the configuration file for the AlphaNet model.
            checkpoint (str): The path to the model checkpoint file.
            device (Literal["cuda", "cpu", "mps"], optional): The device to use for calculations. Defaults to "cpu".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        from alphanet.config import All_Config
        from alphanet.models.model import AlphaNetWrapper

        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # AlphaNet specific attributes
        self.device = device
        self.config = All_Config().from_json(config)
        model = AlphaNetWrapper(self.config.model)
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device(self.device)))
        self.model = model

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """
        Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the AlphaNet potential and other settings
        specified during the initialization of this calculator. The Calculator object is then returned
        to the caller. If the Calculator object has already been created, it is returned directly.

        Returns:
            Calculator: The ASE Calculator object configured with the AlphaNet potential.
        """
        if self._calculator is None:
            from alphanet.infer.calc import AlphaNetCalculator as AlphaNetASECalculator
            self._calculator = AlphaNetASECalculator(
                    model=self.model,
                    device=self.device,
            )
        return self._calculator