"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the EquFlash potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class EquFlashCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the EquFlash potential.

    EquFlash and EquFlashV2 are the MLFF architectures shipped by the `GGNN` package, built on `fairchem-core`
    with e3nn Clebsch-Gordan tensor products accelerated via cuequivariance/FlashTP. `GGNN` publishes pretrained
    checkpoints ("equflash-OAM", "equflash-OMat24", "equflashv2-OAM", "equflashv2-OMat24") that must be downloaded
    manually and passed as `model`. See https://github.com/SamsungDS/GGNN#checkpoints for download links.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stress".
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "",
        device: Literal["cpu", "cuda"] = "cpu",
        seed: int | None = None,
        only_output: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the EquFlashCalculator with the specified model and calculation settings.

        Args:
            model (str, optional): The path to a downloaded EquFlash/EquFlashV2 checkpoint file.
            device (Literal["cuda", "cpu"], optional): The device to use for calculations. Defaults to "cpu".
            seed (int, optional): The seed value for reproducibility. Defaults to None, meaning the seed
                stored in the checkpoint is used.
            only_output (list[str], optional): Restricts the calculator to a subset of the checkpoint's outputs.
                Defaults to None, meaning all of the checkpoint's outputs are kept.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        self.model = model
        self.device = device
        self.seed = seed
        self.only_output = only_output

        self._calculator = None

    @lazy_property("_calculator")
    @requires(
        "GGNN",
        "fairchem",
        hint=(
            'pip install "GGNN @ git+https://github.com/SamsungDS/GGNN.git" installs no dependencies at all. '
            "GGNN's own requirements.txt omits fairchem-core despite hard-importing it. "
            "See docs for the verified dependency set."
        ),
    )
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the EquFlash potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the EquFlash potential.
        """
        from GGNN.common.calculator import UCalculator

        return UCalculator(
            checkpoint_path=self.model,
            cpu=self.device != "cuda",
            seed=self.seed,
            only_output=self.only_output,
        )
