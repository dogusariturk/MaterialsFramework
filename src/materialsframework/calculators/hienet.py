"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the HIENet potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class HIENetCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the HIENet potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stress".

    References:
        - HIENet: https://doi.org/10.48550/arXiv.2503.05771
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(
        self,
        model: str,
        file_type: Literal["checkpoint", "torchscript"] = "checkpoint",
        device: Literal["cuda", "cpu", "mps", "auto"] = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initialize a HIENetCalculator instance with a specified model and calculation settings.

        Args:
            model (str): The path of the HIENet model to use.
            file_type (Literal["checkpoint", "torchscript"], optional): The format of the model file.
                Defaults to 'checkpoint'.
            device (Literal["cuda", "cpu", "mps", "auto"], optional): The device to use for calculations. Defaults to "auto".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Example:
            >>> hienet_calculator = HIENetCalculator(model="checkpoint_600.pth", device="cuda")

        Note:
            The remaining values for the arguments are set to the default values for the HIENet potential.
        """
        super().__init__(**kwargs)

        # HIENet specific attributes
        self.model = model
        self.device = device
        self.file_type = file_type

        self._calculator = None

    @lazy_property("_calculator")
    @requires("hienet", extra="hienet")
    def calculator(self) -> Calculator:
        """Lazily builds and returns the ASE Calculator object for the HIENet potential.

        Returns:
            Calculator: The ASE Calculator object configured with the HIENet potential.
        """
        from hienet.hienet_calculator import HIENetCalculator as HIENetASECalculator

        return HIENetASECalculator(
            model=self.model,
            file_type=self.file_type,
            device=self.device,
        )
