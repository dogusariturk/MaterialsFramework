"""
This module provides a class for performing calculations using the Grace potential.

The `GraceCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and magnetic moments, and to perform structure relaxation using a specified Grace model.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class GraceCalculator(BaseCalculator, BaseMDCalculator):
    """
    A calculator class for performing material property calculations and structure relaxation using the Grace potential.

    The `GraceCalculator` class supports the calculation of properties such as potential energy,
    forces, and stresses. It also allows for the relaxation of structures using a specified Grace model.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute,
                                          including "energy", "forces", and "stress".
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "free_energy", "stress"]

    def __init__(
            self,
            model: str = 'MP_GRACE_2L_r5_4Nov2024',
            pad_neighbors_fraction: float = 0.05,
            pad_atoms_number: int = 1,
            min_dist: float | None = None,
            **kwargs
    ) -> None:
        """
        Initializes the GraceCalculator with the specified model and calculation settings.

        This method sets up the calculator with a predefined Grace model, which will be used
        to calculate properties and perform structure relaxation. Additional parameters
        for the relaxation process can be passed via `basecalculator_kwargs`.

        Args:
            model (str, optional): The Grace model to use. Defaults to 'MP_GRACE_2L_r5_4Nov2024'.
            pad_neighbors_fraction (float, optional): The fraction of neighbors to pad the neighbor list with.
                                                      Defaults to 0.05.
            pad_atoms_number (int, optional): The number of atoms to pad the neighbor list with. Defaults to 1.
            min_dist (float | None, optional): The minimum distance between atoms. Defaults to None.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        from tensorpotential.calculator.foundation_models import grace_fm

        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}
        basemd_kwargs = {key: kwargs.pop(key) for key in BaseMDCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator and BaseMDCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)
        BaseMDCalculator.__init__(self, **basemd_kwargs)

        # Grace specific attributes
        self.model=model
        self.pad_neighbors_fraction=pad_neighbors_fraction
        self.pad_atoms_number=pad_atoms_number
        self.min_dist=min_dist

        self._potential = None
        self._calculator = None

    @property
    def calculator(self) -> Calculator:
        """
        Creates and returns the ASE Calculator object associated with this calculator instance.

        This property initializes the Calculator object using the Grace potential and other settings
        specified during the initialization of this calculator. The Calculator object is then returned
        to the caller. If the Calculator object has already been created, it is returned directly.

        Returns:
            Calculator: The ASE Calculator object configured with the Grace potential.
        """
        if self._calculator is None:
            self._calculator = grace_fm(
                model=self.model,
                pad_neighbors_fraction=self.pad_neighbors_fraction,
                pad_atoms_number=self.pad_atoms_number,
                min_dist=self.min_dist,
            )
        return self._calculator