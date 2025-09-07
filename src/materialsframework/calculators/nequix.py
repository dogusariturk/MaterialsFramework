"""This module provides a class for performing calculations and structure relaxation using the Nequix potential.

The `NequixCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and to perform structure relaxation using a specified Nequix model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class NequixCalculator(BaseCalculator, BaseMDCalculator):
    """A calculator class for performing material property calculations and structure relaxation using the Nequix potential.

    The `NequixCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified Nequix model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stresses".

    References:
        - Nequix: https://arxiv.org/abs/2508.16067
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "nequix-mp-1",
        model_path: str | None = None,
        capacity_multiplier: float = 1.1,
        **kwargs,
    ) -> None:
        """Initializes the NequixCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined Nequix model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model (str): The Nequix model to use.
            model_path (str, optional): The path to the Nequix model to use. Defaults to None.
            capacity_multiplier (float): The multiplier to use for calculating properties. Defaults to 1.1.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # Nequix specific attributes
        self.model = model
        self.model_path = model_path
        self.capacity_multiplier = capacity_multiplier

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the Nequix potential and other settings
        specified during the initialization of this calculator. The Calculator object is then returned
        to the caller. If the Calculator object has already been created, it is returned directly.

        Returns:
            Calculator: The ASE Calculator object configured with the Nequix potential.
        """
        if self._calculator is None:
            from nequix.calculator import NequixCalculator

            self._calculator = NequixCalculator(
                model_name=self.model,
                model_path=self.model_path,
                model_capacity_multiplier=self.capacity_multiplier,
            )
        return self._calculator
