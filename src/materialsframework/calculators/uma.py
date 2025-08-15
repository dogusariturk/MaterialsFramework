"""
This module provides a class for performing calculations and structure relaxation using the UMA potential.

The `UMACalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and to perform structure relaxation using a specified UMA model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class UMACalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the UMA potential.

    The `UMACalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified UMA model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stress".

    References:
        - UMA: https://doi.org/10.48550/arXiv.2506.23971
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
            self,
            model: str = "uma-m-1p1",
            task_name: Literal["omol", "omat", "oc20", "odac", "omc"] = "omat",
            inference_settings: Literal["default", "turbo"] = "default",
            device: Literal["cpu", "cuda"] = "cpu",
            seed: int = 41,
            **kwargs
    ) -> None:
        """
        Initializes the UMACalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined UMA model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # UMA specific attributes
        self.model = model
        self.task_name = task_name
        self.inference_settings = inference_settings
        self.device = device
        self.seed = seed

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """
        Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the UMA potential and other settings
        specified during the initialization of this calculator. The Calculator object is then returned
        to the caller. If the Calculator object has already been created, it is returned directly.

        Returns:
            Calculator: The ASE Calculator object configured with the UMA potential.
        """
        if self._calculator is None:
            from fairchem.core import pretrained_mlip, FAIRChemCalculator
            predictor = pretrained_mlip.get_predict_unit(
                    model_name=self.model,
                    inference_settings=self.inference_settings,
                    device=self.device,
            )
            self._calculator = FAIRChemCalculator(
                    predict_unit=predictor,
                    task_name=self.task_name,
                    seed=self.seed
            )
        return self._calculator
