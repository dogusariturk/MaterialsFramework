"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the ACE potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class ACECalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the ACE potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stress".

    References:
        - ACE: https://doi.org/10.1103/PhysRevB.99.014104
        - ace-jax: https://github.com/ACEsuit/ace-jax
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str,
        **kwargs: Any,
    ) -> None:
        """Initializes the ACECalculator with the specified model and calculation settings.

        Args:
            model (str): Path to an ace-jax model (`.npz`) or a pacemaker PACE potential (`.yace`).
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        self.model = model

        self._calculator = None

    @lazy_property("_calculator")
    @requires("ace_jax", extra="ace")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the ACE potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the ACE potential.
        """
        from ace_jax import ACECalculator as AceJaxASECalculator

        return AceJaxASECalculator(self.model)
