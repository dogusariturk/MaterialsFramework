"""
This module provides a class for performing calculations using the CHGNet potential.

The `CHGNetCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and magnetic moments, and to perform structure relaxation using a specified CHGNet model.
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class CHGNetCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the CHGNet potential.

    The `CHGNetCalculator` class supports the calculation of properties such as potential energy,
    forces, stresses, and magnetic moments. It also allows for the relaxation of structures using
    a specified CHGNet model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", "stress", and "magmoms".

    References:
        - CHGNet: https://doi.org/10.1038/s42256-023-00716-3
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress", "magmoms"]

    def __init__(
            self,
            model: str = "0.3.0",
            stress_weight: float = 1 / 160.21766208,
            include_magmoms: bool = False,
            on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
            device: Literal["cpu", "cuda", "mps"] = "cpu",
            check_cuda_mem: bool = True,
            verbose: bool = False,
            **kwargs
    ) -> None:
        """
        Initializes the CHGNetCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined CHGNet model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters for
        the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model (str, optional): The CHGNet model to use. Defaults to "0.3.0".
            stress_weight (float, optional): Conversion factor for stress from GPa to eV/Å³. Defaults to 1 / 160.21766208.
            include_magmoms (bool, optional): Whether to include magnetic moments in the model. Defaults to False.
            on_isolated_atoms (Literal["ignore", "warn", "error"], optional): Behavior when isolated atoms are detected.
                                                                              Defaults to "warn".
            device (Literal["cpu", "cuda", "mps"], optional): The device to use for calculations. Defaults to "cpu".
            check_cuda_mem (bool, optional): Whether to check CUDA memory before running calculations. Defaults to True.
            verbose (bool, optional): Whether to print verbose output during calculations. Defaults to False.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Examples:
            >>> chgnet_calculator = CHGNetCalculator(model="0.3.0", device="cuda", verbose=True)

        Note:
            The remaining parameters for the CHGNet potential are set to their default values.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, include_magmoms=include_magmoms, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # CHGNet specific attributes
        self.model = model
        self.stress_weight = stress_weight
        self.on_isolated_atoms = on_isolated_atoms
        self.device = device
        self.check_cuda_mem = check_cuda_mem
        self.verbose = verbose

        self._calculator = None
        self._potential = None

    @property
    def potential(self):
        """
        Loads and returns the CHGNet potential associated with this calculator instance.

        This property lazily loads the CHGNet model specified during initialization if it
        has not already been loaded. The loaded potential is then used for all subsequent
        calculations.

        Returns:
            CHGNet: The loaded CHGNet model instance used for calculations.
        """
        if self._potential is None:
            from chgnet.model import CHGNet
            self._potential = CHGNet.load(
                    model_name=self.model,
                    use_device=self.device,
                    check_cuda_mem=self.check_cuda_mem,
                    verbose=self.verbose
            )
        return self._potential

    @property
    def calculator(self) -> Calculator:
        """
        Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the CHGNet potential and other
        relevant attributes such as `device`, `check_cuda_mem`, and `stress_weight`.
        If the Calculator object has already been created, it will return the existing instance.

        Returns:
            Calculator: The ASE Calculator object configured with the CHGNet potential.
        """
        if self._calculator is None:
            from chgnet.model import CHGNetCalculator as CHGNetASECalculator
            self._calculator = CHGNetASECalculator(
                    model=self.potential,
                    use_device=self.device,
                    check_cuda_mem=self.check_cuda_mem,
                    stress_weight=self.stress_weight,
                    on_isolated_atoms=self.on_isolated_atoms
            )
        return self._calculator
