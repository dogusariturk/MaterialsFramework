"""
This module provides a class for performing calculations using the PosEGNN potential.

The `PosEGNNCalculator` class is designed to calculate properties such as potential energy,
forces, and stresses, and to perform structure relaxation using a specified PosEGNN model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class PosEGNNCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the PosEGNN potential.

    The `PosEGNNCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified PosEGNN model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stress".

    References:
        - POS-EGNN: https://github.com/ibm/materials
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
            self,
            model: str,
            device: Literal["cuda", "cpu", "mps"] = "cpu",
            compute_stress: bool = True,
            **kwargs
    ) -> None:
        """
        Initialize a PosEGNNCalculator instance with a specified model and calculation settings.

        Args:
            model (str, optional): The name or the path of the PosEGNN model to use. Defaults to 'PosEGNN-0'.
            device (Literal["cuda", "cpu", "mps"], optional): The device to use for calculations. Defaults to "cpu".
            compute_stress (bool, optional): Whether to compute stress. Defaults to True.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Note:
            The remaining values for the arguments are set to the default values for the PosEGNN potential.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # PosEGNN specific attributes
        self.model = model
        self.device = device
        self.compute_stress = compute_stress

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """
        Returns the ASE calculator associated with this instance.

        If the calculator has not been initialized yet, it will be created
        using the potential attribute of this instance.

        Returns:
            PosEGNNCalculator: The ASE calculator associated with this instance.
        """
        if self._calculator is None:
            from posegnn.calculator import PosEGNNCalculator as PosEGNNASECalculator
            self._calculator = PosEGNNASECalculator(
                    checkpoint=self.model,
                    device=self.device,
                    compute_stress=self.compute_stress,
            )
        return self._calculator
