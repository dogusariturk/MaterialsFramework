"""
This module provides a class for performing calculations using the HIENet potential.

The `HIENetCalculator` class is designed to calculate properties such as potential energy,
forces, and stresses, and to perform structure relaxation using a specified HIENet model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class HIENetCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the HIENet potential.

    The `HIENetCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified HIENet model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stress".

    References:
        - HIENet: https://doi.org/10.48550/arXiv.2503.05771
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(
            self,
            model: str,
            file_type: Literal["checkpoint", "torchscript"] = "checkpoint",
            device: Literal["cuda", "cpu", "mps", "auto"] = "cpu",
            **kwargs
    ) -> None:
        """
        Initialize a HIENetCalculator instance with a specified model and calculation settings.

        Args:
            model (str): The path of the HIENet model to use.
            file_type (Literal["checkpoint", "torchscript"]): The format of the model file.
                Defaults to 'checkpoint'.
            device (Literal["cuda", "cpu", "mps", "auto"], optional): The device to use for calculations. Defaults to "auto".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Example:
            >>> hienet_calculator = HIENetCalculator(model="checkpoint_600.pth", device="cuda")

        Note:
            The remaining values for the arguments are set to the default values for the HIENet potential.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # HIENet specific attributes
        self.model = model
        self.device = device
        self.file_type = file_type

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """
        Returns the ASE calculator associated with this instance.

        If the calculator has not been initialized yet, it will be created
        using the potential attribute of this instance.

        Returns:
            HIENetCalculator: The ASE calculator associated with this instance.
        """
        if self._calculator is None:
            from hienet.hienet_calculator import HIENetCalculator as HIENetASECalculator
            self._calculator = HIENetASECalculator(
                    model=self.model,
                    file_type=self.file_type,
                    device=self.device,
            )
        return self._calculator
