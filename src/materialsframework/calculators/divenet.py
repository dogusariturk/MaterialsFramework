"""
This module provides a class for performing calculations using the DiveNet potential.

The `DiveNetCalculator` class is designed to calculate properties such as potential energy,
forces, and stresses, and to perform structure relaxation using a specified DiveNet model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class DiveNetCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the DiveNet potential.

    The `DiveNetCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified DiveNet model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stress".
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(
            self,
            model: str,
            file_type: Literal["checkpoint", "torchscript"] = "checkpoint",
            device: Literal["cuda", "cpu", "mps", "auto"] = "auto",
            **kwargs
    ) -> None:
        """
        Initialize a DiveNetCalculator instance with a specified model and calculation settings.

        Args:
            model (str): The path of the DiveNet model to use.
            file_type (Literal["checkpoint", "torchscript"]): The format of the model file.
                Defaults to 'checkpoint'.
            device (Literal["cuda", "cpu", "mps", "auto"], optional): The device to use for calculations. Defaults to "auto".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Example:
            >>> divenet_calculator = DiveNetCalculator(model="checkpoint_600.pth", device="cuda")

        Note:
            The remaining values for the arguments are set to the default values for the DiveNet potential.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # DiveNet specific attributes
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
            DiveNetCalculator: The ASE calculator associated with this instance.
        """
        if self._calculator is None:
            from divenet.sevennet_calculator import SevenNetCalculator as DiveNetASECalculator
            self._calculator = DiveNetASECalculator(
                    model=self.model,
                    device=self.device,
                    file_type=self.file_type,
            )
        return self._calculator
