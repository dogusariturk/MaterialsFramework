"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the ALIGNN-FF potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class AlignnCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the ALIGNN-FF potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stresses".

    References:
        - ALIGNN: https://doi.org/10.1038/s41524-021-00650-1
        - ALIGNN-FF: https://doi.org/10.1039/D2DD00096B
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: str | None = None,
        model_filename="best_model.pt",
        config_filename="config.json",
        device: Literal["cuda", "cpu", "mps"] = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initializes the AlignnCalculator with the specified model and calculation settings.

        Args:
            model (str | None, optional): The path to the directory containing the ALIGNN-FF model files. If None,
                'v12.2.2024_dft_3d_307k' model will be used.
            model_filename (str, optional): The filename of the model file. Defaults to "best_model.pt".
            config_filename (str, optional): The filename of the configuration file. Defaults to "config.json".
            device (Literal["cuda", "cpu", "mps"], optional): The device to use for calculations. Defaults to "cpu".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        self.model = model
        self.model_filename = model_filename
        self.config_filename = config_filename
        self.device = device

        self._calculator = None

    @lazy_property("_calculator")
    @requires("alignn", extra="alignn")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the ALIGNN-FF potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the ALIGNN-FF potential.
        """
        from alignn.ff.calculators import AlignnAtomwiseCalculator

        class _AlignnAtomwiseCalculator(AlignnAtomwiseCalculator):
            """AlignnAtomwiseCalculator with `energy` squeezed to an actual scalar."""

            def calculate(self, atoms, properties=None, system_changes=None) -> None:
                super().calculate(atoms, properties, system_changes)
                energy = self.results.get("energy")
                if isinstance(energy, np.ndarray):
                    self.results["energy"] = energy.item()

        return _AlignnAtomwiseCalculator(
            path=self.model,
            model_filename=self.model_filename,
            config_filename=self.config_filename,
            device=self.device,
        )
