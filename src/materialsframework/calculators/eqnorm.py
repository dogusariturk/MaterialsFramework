"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the Eqnorm potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class EqnormCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the Eqnorm potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stresses".
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: Literal["eqnorm-mptrj", "eqnorm-omat", "eqnorm-max-mptrj"] = "eqnorm-mptrj",
        model_name: str = "eqnorm",
        device: str = "cpu",
        compile_model: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the EqnormCalculator with the specified model and calculation settings.

        Args:
            model (Literal["eqnorm-mptrj", "eqnorm-omat", "eqnorm-max-mptrj"]): The Eqnorm model variant. Defaults to "eqnorm-mptrj".
            model_name (str): The name of the Eqnorm model to use for calculations.
            device (str, optional): The device to use for calculations. Defaults to "cpu".
            compile_model (bool, optional): Whether to compile the model with torch.compile. Defaults to False.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        # Eqnorm specific attributes
        self.model = model
        self.model_name = model_name
        self.device = device
        self.compile = compile_model

        self._calculator = None

    @lazy_property("_calculator")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the Eqnorm potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the Eqnorm potential.
        """
        from eqnorm.calculator import EqnormCalculator

        return EqnormCalculator(
            model_variant=self.model,
            model_name=self.model_name,
            device=self.device,
            compile=self.compile,
        )
