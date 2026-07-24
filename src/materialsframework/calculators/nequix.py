"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the Nequix potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class NequixCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the Nequix potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stresses".

    References:
        - Nequix: https://arxiv.org/abs/2508.16067
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "nequix-oam-1",
        model_path: str | None = None,
        capacity_multiplier: float = 1.1,  # Only for jax backend
        backend: str = "jax",
        use_kernel: bool = True,
        use_compile: bool = False,  # Only for torch backend
        **kwargs: Any,
    ) -> None:
        """Initializes the NequixCalculator with the specified model and calculation settings.

        Args:
            model (str, optional): The Nequix model to use. Defaults to "nequix-oam-1".
            model_path (str, optional): The path to the Nequix model to use. Defaults to None.
            capacity_multiplier (float, optional): The multiplier to use for calculating properties. Defaults to 1.1.
            backend (str, optional): The backend to use for calculations. Defaults to "jax".
            use_kernel (bool, optional): Whether to use the kernel for calculations. Defaults to True.
            use_compile (bool, optional): Whether to use compilation for calculations (only applicable for the torch backend). Defaults to False.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        # Nequix specific attributes
        self.model = model
        self.model_path = model_path
        self.capacity_multiplier = capacity_multiplier
        self.backend = backend
        self.use_kernel = use_kernel
        self.use_compile = use_compile

        self._calculator = None

    @lazy_property("_calculator")
    @requires("nequix", extra="nequix")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the Nequix potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the Nequix potential.
        """
        from nequix.calculator import NequixCalculator

        return NequixCalculator(
            model_name=self.model,
            model_path=self.model_path,  # ty: ignore[invalid-argument-type]
            capacity_multiplier=self.capacity_multiplier,
            backend=self.backend,
            use_kernel=self.use_kernel,
            use_compile=self.use_compile,
        )
