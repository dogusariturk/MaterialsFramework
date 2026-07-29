"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the AlphaNet potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class AlphaNetCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the AlphaNet potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stresses".

    References:
        - AlphaNet: https://doi.org/10.48550/arXiv.2501.07155
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str,
        config: str,
        device: Literal["cuda", "cpu", "mps"] = "cpu",
        precision: Literal["32", "64"] = "32",
        **kwargs: Any,
    ) -> None:
        """Initializes the AlphaNetCalculator with the specified model and calculation settings.

        Args:
            model (str): The path to the model checkpoint file.
            config (str): The path to the configuration file for the AlphaNet model.
            device (Literal["cuda", "cpu", "mps"], optional): The device to use for calculations. Defaults to "cpu".
            precision (Literal["32", "64"], optional): The precision of the calculations. Defaults to "32".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        self.model = model
        self.config = config
        self.device = device
        self.precision = precision

        self._calculator = None

    @lazy_property("_calculator")
    @requires("alphanet", extra="alphanet")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the AlphaNet potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the AlphaNet potential.
        """
        from alphanet.config import All_Config
        from alphanet.infer.calc import AlphaNetCalculator as AlphaNetASECalculator

        config = All_Config().from_json(self.config)
        return AlphaNetASECalculator(
            ckpt_path=self.model,
            config=config.model,
            device=self.device,
            precision=self.precision,
        )
