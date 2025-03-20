"""
This module provides a class for performing calculations using the M3GNet potential.

The `M3GNetCalculator` class is designed to calculate properties such as potential energy,
forces, and stresses, and to perform structure relaxation using a specified M3GNet model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class M3GNetCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the M3GNet potential.

    The `M3GNetCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified M3GNet model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "potential_energy", "forces", and "stress".
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
            self,
            model: str = "M3GNet-MP-2021.2.8-PES",
            state_attr: torch.Tensor | None = None,
            stress_weight: float = 1.0,
            device: Literal["cpu", "cuda", "mps"] = "cpu",
            n_cores: int | None = None,
            **kwargs
    ) -> None:
        """
        Initializes the M3GNetCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined M3GNet model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model (str, optional): The M3GNet model to use. Defaults to "M3GNet-MP-2021.2.8-PES".
            state_attr (Tensor | None, optional): State attributes to include in the potential energy calculation.
                                                  This allows for additional model customization. Defaults to None.
            stress_weight (float, optional): Conversion factor from GPa to eV/ang^3. If set to 1.0, stress is calculated in GPa.
                                             Defaults to 1.0.
            device (Literal["cpu", "cuda", "mps"], optional): The device to use for calculations. Defaults to "cpu".
            n_cores (int | None, optional): The number of OMP threads to use. If None, the number of threads is set automatically.
                                            Defaults to None.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Examples:
            >>> m3gnet_calculator = M3GNetCalculator(model="M3GNet-MP-2021.2.8-PES")

        Note:
            The remaining parameters for the M3GNet potential are set to their default values.
        """
        import dgl
        import matgl
        import torch
        from matgl.apps.pes import Potential
        from matgl.ext.ase import PESCalculator

        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # M3GNet specific attributes
        self.model = model
        self.state_attr = state_attr
        self.stress_weight = stress_weight
        self.device = device
        self.n_cores = n_cores

        if self.n_cores is not None:
            dgl.utils.set_num_threads(self.n_cores)

        torch.set_default_device(torch.device(self.device))

        self._calculator = None
        self._potential = None

    @property
    def potential(self) -> Potential:
        """
        Loads and returns the M3GNet potential associated with this calculator instance.

        This property lazily loads the M3GNet model specified during initialization if it
        has not already been loaded. The loaded potential is then used for all subsequent
        calculations.

        Returns:
            Potential: The loaded M3GNet model instance used for calculations.
        """
        if self._potential is None:
            self._potential = matgl.load_model(self.model)
        return self._potential

    @property
    def calculator(self) -> Calculator:
        """
        Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the M3GNet potential and other
        relevant attributes such as `state_attr` and `stress_weight`. If the Calculator object
        has already been created, it will return the existing instance.

        Returns:
            Calculator: The ASE Calculator object configured with the M3GNet potential.
        """
        if self._calculator is None:
            self._calculator = PESCalculator(
                    potential=self.potential,
                    state_attr=self.state_attr,
                    stress_weight=self.stress_weight,
            )
        return self._calculator
