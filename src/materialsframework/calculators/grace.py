"""Calculator for computing potential energy, forces, stresses, and magnetic moments, and relaxing structures with the Grace potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class GraceCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the Grace potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stress".

    References:
        - Grace: https://doi.org/10.1103/PhysRevX.14.021036
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "free_energy", "stress"]

    def __init__(
        self,
        model: str = "GRACE-2L-OMAT",
        pad_neighbors_fraction: float = 0.05,
        pad_atoms_number: int = 1,
        min_dist: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the GraceCalculator with the specified model and calculation settings.

        Args:
            model (str, optional): The Grace model to use. Defaults to 'MP_GRACE_2L_r5_4Nov2024'.
            pad_neighbors_fraction (float, optional): The fraction of neighbors to pad the neighbor list with.
                Defaults to 0.05.
            pad_atoms_number (int, optional): The number of atoms to pad the neighbor list with. Defaults to 1.
            min_dist (float | None, optional): The minimum distance between atoms. Defaults to None.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        self.model = model
        self.pad_neighbors_fraction = pad_neighbors_fraction
        self.pad_atoms_number = pad_atoms_number
        self.min_dist = min_dist

        self._calculator = None

    @lazy_property("_calculator")
    @requires("tensorpotential", extra="grace")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the Grace potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the Grace potential.
        """
        from tensorpotential.calculator.foundation_models import grace_fm

        return grace_fm(
            model=self.model,
            pad_neighbors_fraction=self.pad_neighbors_fraction,
            pad_atoms_number=self.pad_atoms_number,
            min_dist=self.min_dist,
        )
