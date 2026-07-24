"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the UMA potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class UMACalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the UMA potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stress".

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
        **kwargs: Any,
    ) -> None:
        """Initializes the UMACalculator with the specified model and calculation settings.

        Args:
            model (str, optional): The name of the UMA model to use. Defaults to "uma-m-1p1".
            task_name (Literal["omol", "omat", "oc20", "odac", "omc"], optional): The task name. Defaults to "omat".
            inference_settings (Literal["default", "turbo"], optional): The inference settings. Defaults to "default".
            device (Literal["cpu", "cuda"], optional): The device for calculations. Defaults to "cpu".
            seed (int, optional): The seed value for reproducibility. Defaults to 41.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        # UMA specific attributes
        self.model = model
        self.task_name = task_name
        self.inference_settings = inference_settings
        self.device = device
        self.seed = seed

        self._calculator = None

    @lazy_property("_calculator")
    @requires("fairchem", extra="uma")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the UMA potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the UMA potential.
        """
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        predictor = pretrained_mlip.get_predict_unit(
            model_name=self.model,
            inference_settings=self.inference_settings,
            device=self.device,
        )
        return FAIRChemCalculator(predict_unit=predictor, task_name=self.task_name, seed=self.seed)
