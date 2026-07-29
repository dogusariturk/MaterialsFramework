"""Calculator for computing potential energy, free energy, forces, and stresses, and for relaxing structures, with the TACE potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from pathlib import Path

    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class TACECalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the TACE potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "free_energy", "forces", and "stress".

    References:
        - TACE: https://doi.org/10.48550/arXiv.2509.14961
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str | Path = "TACE-v1-OAM-M",
        dtype: str | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
        fidelity_idx: int | None = None,
        target_property: list[str] | None = None,
        neighborlist_backend: Literal["ase", "matscipy", "vesin"] = "matscipy",
        **kwargs: Any,
    ) -> None:
        """Initializes the TACECalculator with the specified model and calculation settings.

        Args:
            model (str | Path, optional): The TACE model to use. This can be the name of a predefined
                foundation model (e.g., "TACE-v1-OAM-M"), which is downloaded and cached automatically, or a
                path to a custom model file ending in ".ckpt", ".pt", ".pth", or ".pt2". Defaults to "TACE-v1-OAM-M".
            dtype (str, optional): The data type to use for the model, e.g. "float32" or "float64". Defaults to
                None, meaning the model's own dtype is used.
            device (Literal["cpu", "cuda"], optional): The device to use for calculations. Defaults to "cpu".
            fidelity_idx (int, optional): The fidelity index to use for multi-fidelity models. Defaults to None,
                meaning the model's default fidelity is used.
            target_property (list[str], optional): The list of properties to predict, overriding the model's
                own target properties. Defaults to None, meaning the model's own target properties are used.
            neighborlist_backend (Literal["ase", "matscipy", "vesin"], optional): The neighbor list backend to
                use. Defaults to "matscipy".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Examples:
            >>> tace_calculator = TACECalculator(model="TACE-v1-OAM-M", device="cuda")

        Note:
            The remaining parameters for the TACE potential are set to their default values.
        """
        super().__init__(**kwargs)

        self.model = model
        self.dtype = dtype
        self.device = device
        self.fidelity_idx = fidelity_idx
        self.target_property = target_property
        self.neighborlist_backend = neighborlist_backend

        self._calculator = None

    @lazy_property("_calculator")
    @requires("tace", extra="tace")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the TACE potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the TACE potential.
        """
        from tace.foundations import tace_foundations
        from tace.interface.ase import TACEAseCalc

        model = tace_foundations.get(self.model, self.model)

        return TACEAseCalc(
            model,
            dtype=self.dtype,
            device=self.device,
            fidelity_idx=self.fidelity_idx,
            target_property=self.target_property,
            neighborlist_backend=self.neighborlist_backend,
        )
