"""This module provides a class for performing calculations and structure relaxation using the NequIP potential.

The `NequIPCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and to perform structure relaxation using a specified NequIP model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class NequIPCalculator(BaseCalculator, BaseMDCalculator):
    """A calculator class for performing material property calculations and structure relaxation using the NequixNequIP potential.

    The `NequIPCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified NequixNequIP model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stresses".

    References:
        - NequIP: https://arxiv.org/abs/2504.16068
    """

    AVAILABLE_PROPERTIES = ["energy", "energies", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "",
        device: Literal["cpu", "cuda"] = "cpu",
        chemical_symbols: list[str] | dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        """Initializes the NequIPCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined NequIP model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model (str): The NequIP model to use.
            device (Literal["cuda", "cpu"]): The device to use for calculations. Defaults to "cpu".
            chemical_symbols (list[str] | dict[str, str] | None): List or mapping of chemical symbols for the system.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # NequIP specific attributes
        self.model = model
        self.device = device
        self.chemical_symbols = chemical_symbols

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the NequIP potential and other settings
        specified during the initialization of this calculator. The Calculator object is then returned
        to the caller. If the Calculator object has already been created, it is returned directly.

        Returns:
            Calculator: The ASE Calculator object configured with the NequIP potential.
        """
        if self._calculator is None:
            from nequip.ase import NequIPCalculator

            self._calculator = NequIPCalculator.from_compiled_model(
                compile_path=self.model, device=self.device, chemical_symbols=self.chemical_symbols
            )

        return self._calculator
