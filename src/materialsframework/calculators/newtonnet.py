"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the NewtonNet potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class NewtonNetCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the NewtonNet potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", "hessian", and "stresses".

    References:
        - NewtonNet: https://doi.org/10.1039/D2DD00008C
    """

    AVAILABLE_PROPERTIES = ["bec", "charges", "energy", "free_energy", "forces", "hessian", "stress"]

    def __init__(
        self,
        model: str | Literal["ani1", "ani1x", "t1x"] = "t1x",
        properties: list | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
        precision: Literal["float64", "float32", "float16"] = "float32",
        **kwargs: Any,
    ) -> None:
        """Initializes the NewtonNetCalculator with the specified model and calculation settings.

        Args:
            model (str | Literal["ani1", "ani1x", "t1x"]): Path to the NewtonNet model or a predefined model name. Defaults to "t1x"
            properties (list): List of properties to calculate, such as "energy", "forces", etc. Defaults to None, which will calculate all available properties.
            device (Literal["cpu", "cuda"]): The device to use for calculations. Defaults to "cpu".
            precision (Literal["float64", "float32", "float16"]): Floating-point precision of the calculations. Defaults to "float32".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        # Cooperatively initializes BaseCalculator, then BaseMDCalculator, per the MRO.
        super().__init__(**kwargs)

        # NewtonNet specific attributes
        if properties is None:
            properties = ["energy", "free_energy", "forces", "hessian", "stress"]
        self.model = model
        self.properties = properties
        self.device = device
        self.precision = precision

        self._calculator = None

    @lazy_property("_calculator")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the NewtonNet potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the NewtonNet potential.
        """
        from newtonnet.utils.ase_interface import (
            MLAseCalculator as NewtonNetASECalculator,
        )

        return NewtonNetASECalculator(
            model_path=self.model,
            properties=self.properties,
            device=self.device,
            precision=self.precision,
        )
