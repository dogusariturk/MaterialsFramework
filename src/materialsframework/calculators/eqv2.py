"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the EqV2 potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class EqV2Calculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the EqV2 potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stress".

    References:
        - eqv2: https://doi.org/10.48550/arXiv.2410.12771
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "EquiformerV2-153M-OMAT24-MP-sAlex",
        checkpoint_path: str | None = None,
        local_cache: str = "~/.cache/eqv2/",
        device: Literal["cpu", "cuda"] = "cpu",
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the EqV2Calculator with the specified model and calculation settings.

        Args:
            model (str, optional): The name of the EqV2 model to use for calculations. Defaults to 'EquiformerV2-153M-OMAT24-MP-sAlex'.
            checkpoint_path (str, optional): The path to the checkpoint file for the EqV2 model.
            local_cache (str, optional): The path to the local cache directory for the EqV2 model. Defaults to "~/.cache/eqv2/".
            device (Literal["cpu", "cuda"], optional): The device to use for the calculations. Defaults to "cpu".
            seed (int, optional): The seed value for the model.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        self.model = model
        self.checkpoint_path = checkpoint_path
        self.local_cache = local_cache
        self.device = device
        self.seed = seed

        self._calculator = None

    @lazy_property("_calculator")
    @requires("fairchem", extra="eqv2")
    def calculator(self) -> Calculator:
        """Creates and returns the ASE Calculator object associated with this calculator instance.

        Returns:
            Calculator: The ASE Calculator object configured with the EqV2 potential.
        """
        from fairchem.core import OCPCalculator

        return OCPCalculator(
            model_name=self.model,
            checkpoint_path=self.checkpoint_path,
            local_cache=self.local_cache,
            cpu=self.device != "cuda",
            seed=self.seed,
        )
