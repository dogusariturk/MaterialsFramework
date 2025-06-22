"""
This module provides a class for performing calculations and structure relaxation using the PET-MAD potential.

The `PetMadCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and to perform structure relaxation using a specified PET-MAD model.
"""
from __future__ import annotations

from typing import Literal, Optional, TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class PetMadCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the PET-MAD potential.

    The `PetMadCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified PET-MAD model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stresses".

    References:
        - PET-MAD: https://doi.org/10.48550/arXiv.2503.14118
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "1.0.1",
        checkpoint_path: Optional[str] = None,
        device: Literal["cuda", "cpu", "mps"] = "cpu",
        **kwargs,
    ) -> None:
        """
        Initializes the PetMadCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined PET-MAD model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model (str): The version of the PET-MAD model to use. Default is "1.0.1".
            checkpoint_path (str, optional): Path to the model checkpoint file. If not provided,
                                                the model will be downloaded using the "version" parameter.
            device (str): The device to use for calculations. Options are "cuda", "cpu", or "mps".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # PET-MAD specific attributes
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.device = device

        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """
        Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the PET-MAD potential and other settings
        specified during the initialization of this calculator. The Calculator object is then returned
        to the caller. If the Calculator object has already been created, it is returned directly.

        Returns:
            Calculator: The ASE Calculator object configured with the PET-MAD potential.
        """
        if self._calculator is None:
            from pet_mad.calculator import PETMADCalculator
            self._calculator = PETMADCalculator(
                version=self.model,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
            )
        return self._calculator
