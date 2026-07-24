"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the PosEGNN potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class PosEGNNCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the PosEGNN potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stress".

    References:
        - POS-EGNN: https://github.com/ibm/materials
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: str,
        device: Literal["cuda", "cpu", "mps"] = "cpu",
        compute_stress: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize a PosEGNNCalculator instance with a specified model and calculation settings.

        Args:
            model (str): The name or the path of the PosEGNN model to use.
            device (Literal["cuda", "cpu", "mps"], optional): The device to use for calculations. Defaults to "cpu".
            compute_stress (bool, optional): Whether to compute stress. Defaults to True.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Note:
            The remaining values for the arguments are set to the default values for the PosEGNN potential.
        """
        super().__init__(**kwargs)

        # PosEGNN specific attributes
        self.model = model
        self.device = device
        self.compute_stress = compute_stress

        self._calculator = None

    @lazy_property("_calculator")
    @requires(
        "posegnn",
        hint="clone https://github.com/IBM/materials and add models/pos_egnn to PYTHONPATH (see the posegnn job in .github/workflows/tests.yml)",
    )
    def calculator(self) -> Calculator:
        """Lazily builds and returns the ASE Calculator object for the PosEGNN potential.

        Returns:
            Calculator: The ASE Calculator object configured with the PosEGNN potential.
        """
        from posegnn.calculator import PosEGNNCalculator as PosEGNNASECalculator

        return PosEGNNASECalculator(
            checkpoint=self.model,
            device=self.device,
            compute_stress=self.compute_stress,
        )
