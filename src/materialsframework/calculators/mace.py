"""Calculator for computing potential energy, free energy, forces, and stresses, and for relaxing structures, with the MACE potential."""

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


class MACECalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the MACE potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "free_energy", "forces", and "stress".

    References:
        - MACE: https://doi.org/10.48550/arXiv.2401.00096
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "node_energy", "forces", "stress"]

    def __init__(
        self,
        model: str | Path = "medium-mpa-0",
        head: str | None = None,
        include_dipoles: bool = False,
        device: Literal["cuda", "cpu", "mps"] = "cpu",
        default_dtype: str = "",
        model_type: Literal["MACE", "DipoleMACE", "EnergyDipoleMACE"] = "MACE",
        **kwargs: Any,
    ) -> None:
        """Initializes the MACECalculator with the specified model and calculation settings.

        Args:
            model (str | Path, optional): The MACE model to use. This can be the name of a predefined model
                (e.g., "medium-omat-0"), a path to a custom model file, or a URL. Defaults to "medium-mpa-0".
            head (str | None, optional): The head of the model file. Defaults to None.
            include_dipoles (bool, optional): Determines whether dipole properties are included in the model. Defaults to False.
            device (Literal["cuda", "cpu", "mps"], optional): The device to use for calculations. Defaults to "cpu".
            default_dtype (str, optional): The default data type to be used for the model. Defaults to an empty string,
                meaning the default data type of the model will be used.
            model_type (Literal["MACE", "DipoleMACE", "EnergyDipoleMACE"], optional): The type of MACE model to use. Defaults to "MACE".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Examples:
            >>> mace_calculator = MACECalculator(model="large", device="cuda")

        Note:
            The remaining parameters for the MACE potential are set to their default values.
        """
        super().__init__(include_dipoles=include_dipoles, **kwargs)

        # MACE specific attributes
        self.model = model
        self.head = head
        self.device = device
        self.default_dtype = default_dtype
        self.model_type = model_type

        if include_dipoles:
            self.AVAILABLE_PROPERTIES = [*MACECalculator.AVAILABLE_PROPERTIES, "dipole"]

        self._calculator = None

    @lazy_property("_calculator")
    @requires("mace", extra="mace")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the MACE potential, using `device`, `default_dtype`, and `model_type`.

        Returns:
            Calculator: The ASE Calculator object configured with the MACE potential.
        """
        from mace.calculators import mace_mp

        return mace_mp(
            model=self.model,
            device=self.device,
            default_dtype=self.default_dtype,
            model_type=self.model_type,
            head=self.head,
        )
