"""This module provides a class for calculations and relaxations with PET-MAD models via UPET."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property

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
        **kwargs,
    ) -> None:
        """Initializes the PetMadCalculator with the specified model and calculation settings.

        Args:
            model (str): PET-MLIP model to use. Default is "pet-mad-s". Ignored if `checkpoint_path` is provided.
            version (str): Version of the model to use. Default is "latest". Ignored if `checkpoint_path` is provided.
            checkpoint_path (str, optional): Path to the model checkpoint file. If not provided, the model will be
                downloaded using the "version" parameter.
            device (str): The device to use for calculations. Options are "cuda", "cpu", or "mps".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        # PET-MAD via UPET specific attributes
        self.model = model
        self.version = version
        self.checkpoint_path = checkpoint_path
        self.device = device

        self._calculator = None

    @lazy_property("_calculator")
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
