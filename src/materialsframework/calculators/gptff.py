"""
This module provides a class for performing calculations and structure relaxation using the GPTFF potential.

The `GPTFFCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and to perform structure relaxation using a specified GPTFF model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class GPTFFCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the GPTFF potential.

    The `GPTFFCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified GPTFF model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stresses".
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
            self,
            model_path: str,
            device: Literal["cpu", "cuda"] = "cpu",
            **kwargs
    ) -> None:
        """
        Initializes the GPTFFCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined GPTFF model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model_path (str): Path to the GPTFF model file.
            device (Literal["cpu", "cuda"]): Device to use for calculations ("cpu" or "cuda").
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # GPTFF specific attributes
        self.model_path = model_path
        self.device = device

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """
        Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the GPTFF potential and other settings
        specified during the initialization of this calculator. The Calculator object is then returned
        to the caller. If the Calculator object has already been created, it is returned directly.

        Returns:
            Calculator: The ASE Calculator object configured with the GPTFF potential.
        """
        if self._calculator is None:
            from gptff.model.mpredict import ASECalculator as GPTFFASECalculator
            self._calculator = GPTFFASECalculator(
                    model_path=self.model_path,
                    device=self.device
            )
        return self._calculator