"""
This module provides a class for performing calculations and structure relaxation using the DeePMD potential.

The `DeePMDCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and magnetic moments, and to perform structure relaxation using a specified DeePMD model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING, Union

from deepmd.calculator import DP

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from pathlib import Path

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class DeePMDCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the DeePMD potential.

    The `DeePMDCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified DeePMD model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stress".
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress", "virial"]

    def __init__(
            self,
            model: Union[str, Path],
            **kwargs
    ) -> None:
        """
        Initializes the DeePMDCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined DeePMD model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # DeePMD specific attributes
        self.model = model

        self._potential = None
        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """
        Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the DeePMD potential and other settings
        specified during the initialization of this calculator. The Calculator object is then returned
        to the caller. If the Calculator object has already been created, it is returned directly.

        Returns:
            Calculator: The ASE Calculator object configured with the DeePMD potential.
        """
        if self._calculator is None:
            self._calculator = DP(model=self.model)
        return self._calculator