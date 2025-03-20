"""
This module provides a class for performing calculations using the MatterSim potential.

The `MatterSimCalculator` class is designed to calculate properties such as potential energy,
forces, and stresses, and to perform structure relaxation using a specified MatterSim model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from mattersim.forcefield import MatterSimCalculator as MatterSimASECalculator, Potential

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class MatterSimCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the MatterSim potential.

    The `MatterSimCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified MatterSim model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stress".
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "stress"]

    def __init__(
            self,
            model = 'mattersim-v1.0.0-5m',
            args_dict: dict | None = None,
            compute_stress: bool = True,
            stress_weight: float = 1.0,
            device: Literal["cuda", "cpu"] = "cpu",
            **kwargs
    ) -> None:
        """
        Initializes the MatterSimCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined MatterSim model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `kwargs`.

        Args:
            model (str, optional): The name of the MatterSim model to use. Defaults to 'mattersim-v1.0.0-5m'.
            args_dict (dict, optional): A dictionary of additional arguments to pass to the MatterSim calculator.
                Defaults to None.
            compute_stress (bool, optional): Whether to compute the stress tensor. Defaults to True.
            stress_weight (float, optional): Conversion factor from GPa to eV/ang^3. If set to 1.0, stress is calculated in GPa.
                                             Defaults to 1.0.
            device (Literal["cuda", "cpu"], optional): The device to use for calculations. Defaults to "cpu".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Examples:
            >>> mattersim_calculator = MatterSimCalculator(model="mattersim-v1.0.0-5m", device="cuda")

        Note:
            The remaining parameters for the MatterSim potential are set to their default values.
        """

        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # MatterSim specific attributes
        self.model = model
        self.args_dict = args_dict or {}
        self.compute_stress = compute_stress
        self.stress_weight = stress_weight
        self.device = device

        self._calculator = None
        self._potential = None

    @property
    def potential(self) -> Potential:
        """
        Loads and returns the MatterSim potential associated with this calculator instance.

        This property lazily loads the MatterSim model specified during initialization if it
        has not already been loaded. The loaded potential is then used for all subsequent
        calculations.

        Returns:
            Potential: The loaded MatterSim model instance used for calculations.
        """
        if self._potential is None:
            self._potential = Potential.from_checkpoint(
                    load_path=self.model,
                    device=self.device
            )
        return self._potential

    @property
    def calculator(self) -> Calculator:
        """
        Creates and returns the ASE Calculator object associated with this calculator instance.

        This property creates an ASE Calculator object configured with the MatterSim potential
        and the specified calculation settings. The calculator is then used for all subsequent
        calculations.

        Returns:
            Calculator: The ASE Calculator object configured with the MatterSim potential.
        """
        if self._calculator is None:
            self._calculator = MatterSimASECalculator(
                    potential=self.potential,
                    args_dict=self.args_dict,
                    compute_stress=self.compute_stress,
                    stress_weight=self.stress_weight,
                    device=self.device,
            )
        return self._calculator
