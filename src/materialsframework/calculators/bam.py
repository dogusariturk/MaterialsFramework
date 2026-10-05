"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the BAM potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class BAMCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the BAM potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stress".

    References:
        - BAM: https://doi.org/10.1038/s41524-026-02258-9
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str,
        device: Literal["cpu", "cuda"] = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initializes the BAMCalculator with the specified model and calculation settings.

        Args:
            model (str): Path to a trained BAM model checkpoint (`.pkl`).
            device (Literal["cpu", "cuda"], optional): The device to use for calculations. Defaults to "cpu".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        self.model = model
        self.device = device

        self._calculator = None

    @lazy_property("_calculator")
    @requires("bam_torch", hint='pip install "BAM-torch @ git+https://github.com/myung-group/BAM-torch.git"')
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the BAM potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the BAM potential.
        """
        from bam_torch.tase.base_calculator import RACECalculator

        return RACECalculator(model=self.model, device=self.device)
