"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the eSEN potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class eSENCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the eSEN potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stress".

    References:
        - eSEN: https://doi.org/10.48550/arXiv.2502.12147
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "esen-sm-conserving-all-omol",
        checkpoint_path: str | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initializes the eSENCalculator with the specified model and calculation settings.

        Args:
            model (str): The name of the eSEN model to use for calculations. Must be one of the
                models available in fairchem-core (e.g. ``esen-sm-conserving-all-omol``,
                ``esen-md-direct-all-omol``, ``esen-sm-conserving-all-oc25``). Ignored if
                ``checkpoint_path`` is provided. Defaults to ``"esen-sm-conserving-all-omol"``.
            checkpoint_path (str, optional): Path to a local eSEN checkpoint file. When provided,
                the model registry name is ignored.
            device (Literal["cpu", "cuda"]): The device to use for the calculations. Defaults to "cpu".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        # eSEN specific attributes
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.device = device

        self._calculator = None

    @lazy_property("_calculator")
    def calculator(self) -> Calculator:
        """Creates and returns the ASE Calculator object associated with this calculator instance.

        Returns:
            Calculator: The ASE Calculator object configured with the eSEN potential.
        """
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        from fairchem.core.units.mlip_unit import load_predict_unit

        if self.checkpoint_path is not None:
            predictor = load_predict_unit(path=self.checkpoint_path, device=self.device)
        else:
            predictor = pretrained_mlip.get_predict_unit(model_name=self.model, device=self.device)
        return FAIRChemCalculator(predict_unit=predictor)
