"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the Prophet potential."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class ProphetCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the Prophet potential.

    The matching upstream ASE calculator is selected from the checkpoint's own JSON header. Checkpoints with spin
    conditioning additionally take each atom's local magnetic moment as an input, read from the structure's initial
    magnetic moments. Moments that are not set are passed as zeros, i.e. a non-magnetic configuration.

    With a magnetic-moment prediction bundle passed as `magmom_head`, the initial magnetic moments instead act as seeds
    encoding the magnetic ordering: each atom's moment is predicted first, energy, forces, and stress are evaluated at
    the predicted moments, and the predicted moments are returned as "magmoms" and recorded in the relaxation trajectory.
    Seeds must be collinear and not all zero.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stress", plus "magmoms" when `magmom_head` is given.

    References:
        - Prophet: https://www.kairosmaterials.com/papers/Prophet.pdf
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str,
        magmom_head: str | None = None,
        seed_mode: Literal["lut", "raw"] = "lut",  # Only with magmom_head
        magnetic_threshold: float = 0.5,  # Only with magmom_head
        device: Literal["cpu", "cuda"] = "cpu",
        use_kernel: bool = False,
        use_compile: bool = False,  # Only for checkpoints without spin conditioning
        **kwargs: Any,
    ) -> None:
        """Initializes the ProphetCalculator with the specified model and calculation settings.

        Args:
            model (str): Path to a Prophet checkpoint file, such as one from https://huggingface.co/kairosmaterial/prophet.
            magmom_head (str | None, optional): Path to a Prophet magnetic-moment prediction bundle. If given, `model` must
                be a spin-conditioned checkpoint, and each atom's magnetic moment is predicted before evaluating energy,
                forces, and stress at the predicted moments. Defaults to None.
            seed_mode (Literal["lut", "raw"], optional): How seed moments become the predictor's starting moments (only
                applicable with `magmom_head`). "lut" keeps each seed's sign for atoms whose seed magnitude exceeds
                `magnetic_threshold`, takes the magnitude from the bundle's per-element lookup table, and zeroes the rest;
                "raw" uses the seeds as given. Defaults to "lut".
            magnetic_threshold (float, optional): Seed magnitude above which an atom counts as magnetic in "lut" seed mode
                (only applicable with `magmom_head`). Defaults to 0.5.
            device (Literal["cpu", "cuda"], optional): The device to use for calculations. Defaults to "cpu".
            use_kernel (bool, optional): Whether to use the OpenEquivariance CUDA tensor-product kernels instead of the
                pure e3nn path. Requires a CUDA device. Defaults to False.
            use_compile (bool, optional): Whether to compile the model with `torch.compile` (only applicable to
                checkpoints without spin conditioning, on a CUDA device). Defaults to False.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        if magmom_head is not None:
            self.AVAILABLE_PROPERTIES = [*ProphetCalculator.AVAILABLE_PROPERTIES, "magmoms"]
            kwargs.setdefault("include_magmoms", True)

        super().__init__(**kwargs)

        self.model = model
        self.magmom_head = magmom_head
        self.seed_mode = seed_mode
        self.magnetic_threshold = magnetic_threshold
        self.device = device
        self.use_kernel = use_kernel
        self.use_compile = use_compile

        self._calculator = None

    @lazy_property("_calculator")
    @requires("prophet", "prophet_spin", hint='pip install "prophet-mlip @ git+https://github.com/kairosmaterial/prophet.git"')
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the Prophet potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the Prophet potential.

        Raises:
            ValueError: If `magmom_head` is given but `model` is not a spin-conditioned checkpoint.
        """
        from ase.calculators.calculator import all_changes

        with Path(self.model).open("rb") as f:
            config = json.loads(f.readline())

        if self.magmom_head is not None:
            if not config.get("magmom_conditioning"):
                raise ValueError(
                    f"`magmom_head` requires a spin-conditioned `model` checkpoint, but {self.model} has no spin conditioning."
                )

            from prophet_spin import MagmomPredictor

            return MagmomPredictor(
                model_path=self.model,
                head_path=self.magmom_head,
                device=self.device,
                seed_mode=self.seed_mode,
                magnetic_threshold=self.magnetic_threshold,
                use_kernel=self.use_kernel,
            )

        if config.get("magmom_conditioning"):
            from prophet_spin import SpinCalculator

            return SpinCalculator(model_path=self.model, device=self.device, use_kernel=self.use_kernel)

        from prophet import KairosCalculator

        class _KairosCalculator(KairosCalculator):
            """KairosCalculator with `energy`/`free_energy` squeezed to actual scalars."""

            def calculate(self, atoms=None, properties=None, system_changes=all_changes) -> None:
                super().calculate(atoms, properties, system_changes)
                for name in ("energy", "free_energy"):
                    value = self.results.get(name)
                    if isinstance(value, np.ndarray):
                        self.results[name] = value.item()

        return _KairosCalculator(
            model_path=self.model,
            use_kernel=self.use_kernel,
            use_compile=self.use_compile,
            device=self.device,
        )
