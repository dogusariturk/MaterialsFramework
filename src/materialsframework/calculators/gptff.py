"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the GPTFF potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class GPTFFCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the GPTFF potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stresses".

    References:
        - GPTFF: https://doi.org/10.1016/j.scib.2024.08.039
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "GPTFF-MatPES_PBE_2025.2",
        device: Literal["cpu", "cuda"] = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initializes the GPTFFCalculator with the specified model and calculation settings.

        Args:
            model (str, optional): Path to the GPTFF model file. Defaults to "GPTFF-MatPES_PBE_2025.2".
            device (Literal["cpu", "cuda"], optional): Device to use for calculations ("cpu" or "cuda"). Defaults to "cpu".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        # GPTFF-specific attributes
        self.model = model
        self.device = device

        self._calculator = None

    @lazy_property("_calculator")
    @requires("gptff", hint='pip install "gptff @ git+https://github.com/atomly-materials-research-lab/GPTFF.git"')
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the GPTFF potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the GPTFF potential.
        """
        from gptff.interfaces import ASECalculator as GPTFFASECalculator

        return GPTFFASECalculator(model_path=self.model, device=self.device)
