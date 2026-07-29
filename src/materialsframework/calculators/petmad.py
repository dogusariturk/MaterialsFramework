"""Calculator for computing potential energy, forces, and stresses, and relaxing structures with the PET-MAD potential, served via UPET."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class PetMadCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using PET-MAD models served by UPET.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stresses".

    References:
        - UPET: https://github.com/lab-cosmo/upet
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "pet-mad-s",
        version: str = "latest",
        checkpoint_path: str | None = None,
        device: Literal["cuda", "cpu", "mps"] = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initializes the PetMadCalculator with the specified model and calculation settings.

        Args:
            model (str, optional): PET-MLIP model to use. Default is "pet-mad-s". Ignored if `checkpoint_path` is provided.
            version (str, optional): Version of the model to use. Default is "latest". Ignored if `checkpoint_path` is provided.
            checkpoint_path (str, optional): Path to the model checkpoint file. If not provided, the model will be
                downloaded using the "version" parameter.
            device (Literal["cuda", "cpu", "mps"], optional): The device to use for calculations. Defaults to "cpu".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        self.model = model
        self.version = version
        self.checkpoint_path = checkpoint_path
        self.device = device

        self._calculator = None

    @lazy_property("_calculator")
    @requires("upet", extra="petmad")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the PET-MAD model, using UPET.

        Returns:
            Calculator: The ASE Calculator object configured via UPET.
        """
        from upet.calculator import UPETCalculator

        return UPETCalculator(
            model=self.model,
            version=self.version,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
