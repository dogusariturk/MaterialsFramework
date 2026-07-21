"""Calculator for computing potential energy, forces, stresses, and magnetic moments, and relaxing structures with the MatRIS potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class MatRISCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the MatRIS potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", "stress", and "magmoms".

    References:
        - MatRIS: https://doi.org/10.48550/arXiv.2603.02002
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress", "magmoms"]

    def __init__(
        self,
        model: Literal["matris_10m_oam", "matris_10m_mp"] = "matris_10m_oam",
        task: Literal["e", "em", "ef", "efs", "efsm"] = "efsm",
        device: Literal["cpu", "cuda"] = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initializes the MatRISCalculator with the specified model and calculation settings.

        Args:
            model (Literal["matris_10m_oam", "matris_10m_mp"], optional): The MatRIS foundation model to use.
                "matris_10m_oam" is trained on OMat24 and fine-tuned on sAlex+MPtrj, while "matris_10m_mp" is
                trained on MPtrj. Weights are downloaded and cached automatically. Defaults to "matris_10m_oam".
            task (Literal["e", "em", "ef", "efs", "efsm"], optional): The prediction task, selecting which of
                energy ("e"), magnetic moments ("m"), forces ("f"), and stress ("s") the model predicts. Defaults
                to "efsm".
            device (Literal["cpu", "cuda"], optional): The device to use for calculations. Defaults to "cpu".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Examples:
            >>> matris_calculator = MatRISCalculator(model="matris_10m_oam", device="cuda")

        Note:
            The remaining parameters for the MatRIS potential are set to their default values.
        """
        super().__init__(**kwargs)

        # MatRIS specific attributes
        self.model = model
        self.task = task
        self.device = device

        self._calculator = None

    @lazy_property("_calculator")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the MatRIS potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the MatRIS potential.
        """
        from matris.applications.base import MatRISCalculator as _MatRISCalculator

        return _MatRISCalculator(
            model=self.model,
            task=self.task,
            device=self.device,
        )
