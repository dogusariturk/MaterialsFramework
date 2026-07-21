"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the ORB potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class ORBCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the ORB potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stress".

    References:
        - Orb-v3: https://doi.org/10.48550/arXiv.2504.06231
        - Orb-v2: https://doi.org/10.48550/arXiv.2410.22570
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "orb-v3-conservative-inf-omat",
        device: Literal["cuda", "cpu", "mps"] = "cpu",
        precision: Literal["float32-high", "float32-highest", "float64"] = "float32-high",
        **kwargs: Any,
    ) -> None:
        """Initializes the ORBCalculator with the specified model and calculation settings.

        Args:
            model (str, optional): The name of the ORB model to use. Defaults to "orb-v3-conservative-inf-omat".
            device (Literal["cuda", "cpu", "mps"], optional): The device to use for calculations. Defaults to "cpu".
            precision (Literal["float32-high", "float32-highest", "float64"], optional): The floating point precision to use for the model.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Examples:
            >>> orb_calculator = ORBCalculator(model="orb-v2", device="cuda")
        """
        super().__init__(**kwargs)

        # ORB specific attributes
        self.model = model
        self.device = device
        self.precision = precision

        self._calculator = None

    @lazy_property("_calculator")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the ORB potential.

        Returns:
            Calculator: The ASE Calculator object configured with the ORB potential.
        """
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import (
            ORBCalculator as ORBASECalculator,
        )

        model = pretrained.ORB_PRETRAINED_MODELS[self.model]
        potential, atoms_adapter = model(device=self.device, precision=self.precision)  # ty:ignore[unknown-argument]

        return ORBASECalculator(
            potential,
            atoms_adapter,
            device=self.device,
        )
