"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the DeePMD potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from pathlib import Path

    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class DeePMDCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the DeePMD potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stress".

    References:
        - DeePMD-kit: https://doi.org/10.1016/j.cpc.2018.03.016
        - DeePMD-kit v2: https://doi.org/10.1063/5.0155600
        - DeePMD-kit v3: https://doi.org/10.48550/arXiv.2502.19161
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "virial", "stress"]

    def __init__(self, model: str | Path, **kwargs: Any) -> None:
        """Initializes the DeePMDCalculator with the specified model and calculation settings.

        Args:
            model (str | Path): The path to the DeePMD model file.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        self.model = model

        self._calculator = None

    @lazy_property("_calculator")
    @requires("deepmd", extra="deepmd")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the DeePMD potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the DeePMD potential.
        """
        from deepmd.calculator import DP

        return DP(model=self.model)
