"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the SevenNet potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class SevenNetCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the SevenNet potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stress".

    References:
        - SevenNet: https://doi.org/10.1021/acs.jctc.4c00190
        - SevenNet-MF: https://doi.org/10.1021/jacs.4c14455
    """

    AVAILABLE_PROPERTIES = ["energy", "energies", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "7net-omni",
        modal: str = "mpa",
        file_type: Literal["checkpoint", "torchscript"] = "checkpoint",
        device: Literal["cuda", "cpu", "mps", "auto"] = "auto",
        **kwargs: Any,
    ) -> None:
        """Initialize a SevenNetCalculator instance with a specified model and calculation settings.

        Args:
            model (str, optional): The name or the path of the SevenNet model to use. Defaults to '7net-mf-ompa'.
            modal (Literal["mpa", "omat24"], optional): The fidelity of the model to use. Defaults to 'mpa'.
            file_type (Literal["checkpoint", "torchscript"]): The format of the model file.
                Defaults to 'checkpoint'.
            device (Literal["cuda", "cpu", "mps", "auto"], optional): The device to use for calculations. Defaults to "auto".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Example:
            >>> sevennet_calculator = SevenNetCalculator(model="SevenNet-0", device="cuda")

        Note:
            The remaining values for the arguments are set to the default values for the SevenNet potential.
        """
        super().__init__(**kwargs)

        # SevenNet specific attributes
        self.model = model
        self.modal = modal
        self.device = device
        self.file_type = file_type

        self._calculator = None

    @lazy_property("_calculator")
    def calculator(self) -> Calculator:
        """Lazily builds and returns the ASE Calculator object for the SevenNet potential.

        Returns:
            Calculator: The ASE Calculator object configured with the SevenNet potential.
        """
        from sevenn.calculator import SevenNetCalculator as SevenNetASECalculator

        return SevenNetASECalculator(
            model=self.model,
            modal=self.modal,
            device=self.device,
            file_type=self.file_type,
        )
