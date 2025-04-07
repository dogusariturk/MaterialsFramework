"""
This module provides a class for performing calculations using the ORB potential.

The `ORBCalculator` class is designed to calculate properties such as potential energy,
forces, and stresses, and to perform structure relaxation using a specified ORB model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class ORBCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the ORB potential.

    The `ORBCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified ORB model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stress".
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "stress"]

    def __init__(
            self,
            model: str = 'orb-v2',
            device: Literal["cuda", "cpu", "mps"] = "cpu",
            brute_force_knn: bool | None = None,
            **kwargs
    ) -> None:
        """
        Initializes the ORBCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined ORB model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model (str, optional): The name of the ORB model to use. Defaults to 'orb-v2'.
            device (Literal["cuda", "cpu", "mps"], optional): The device to use for calculations. Defaults to "cpu".
            brute_force_knn (bool, optional): Whether to use brute-force k-nearest neighbors search.
                Defaults to None.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Examples:
            >>> orb_calculator = ORBCalculator(model="orb-v2", device="cuda")
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # ORB specific attributes
        self.model = model
        self.device = device
        self.brute_force_knn = brute_force_knn

        self._potential = None
        self._calculator = None

    @property
    def potential(self):
        """
        Loads and returns the ORB potential associated with this calculator instance.

        This property lazily loads the ORB model specified during initialization if it
        has not already been loaded. The loaded potential is then used for all subsequent
        calculations.

        Returns:
            GraphRegressor: The ORB potential associated with this instance.
        """
        if self._potential is None:
            from orb_models.forcefield import pretrained
            model = pretrained.ORB_PRETRAINED_MODELS[self.model]
            self._potential = model(device=self.device)
        return self._potential

    @property
    def calculator(self) -> Calculator:
        """
        Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the ORB potential and other
        relevant attributes such as `device`, `brute_force_knn`, and `system_config`. If the Calculator
        object has already been created, it will return the existing instance.

        Returns:
            Calculator: The ASE Calculator object configured with the ORB potential.
        """
        if self._calculator is None:
            from orb_models.forcefield.calculator import ORBCalculator as ORBASECalculator
            self._calculator = ORBASECalculator(
                    model=self.potential,
                    device=self.device,
                    brute_force_knn=self.brute_force_knn,
            )
        return self._calculator
