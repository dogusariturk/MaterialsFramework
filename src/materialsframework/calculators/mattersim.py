"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the MatterSim potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from mattersim.forcefield import Potential

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class MatterSimCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the MatterSim potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stress".

    References:
        - MatterSim: https://doi.org/10.48550/arXiv.2405.04967
    """

    AVAILABLE_PROPERTIES = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "mattersim-v1.0.0-5m",
        args_dict: dict | None = None,
        compute_stress: bool = True,
        stress_weight: float = 1.0,
        device: Literal["cuda", "cpu"] = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initializes the MatterSimCalculator with the specified model and calculation settings.

        Args:
            model (str, optional): The name of the MatterSim model to use. Defaults to 'mattersim-v1.0.0-5m'.
            args_dict (dict, optional): A dictionary of additional arguments to pass to the MatterSim calculator.
                Defaults to None.
            compute_stress (bool, optional): Whether to compute the stress tensor. Defaults to True.
            stress_weight (float, optional): Conversion factor from GPa to eV/ang^3. If set to 1.0, stress is
                calculated in GPa. Defaults to 1.0.
            device (Literal["cuda", "cpu"], optional): The device to use for calculations. Defaults to "cpu".
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Examples:
            >>> mattersim_calculator = MatterSimCalculator(model="mattersim-v1.0.0-5m", device="cuda")

        Note:
            The remaining parameters for the MatterSim potential are set to their default values.
        """
        super().__init__(**kwargs)

        # MatterSim specific attributes
        self.model = model
        self.args_dict = args_dict or {}
        self.compute_stress = compute_stress
        self.stress_weight = stress_weight
        self.device = device

        self._calculator = None
        self._potential = None

    @lazy_property("_potential")
    def potential(self) -> Potential:
        """Lazily loads and returns the MatterSim potential specified during initialization.

        Returns:
            Potential: The loaded MatterSim model instance used for calculations.
        """
        from mattersim.forcefield import Potential

        return Potential.from_checkpoint(load_path=self.model, device=self.device)

    @lazy_property("_calculator")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the MatterSim potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the MatterSim potential.
        """
        from mattersim.forcefield import (
            MatterSimCalculator as MatterSimASECalculator,
        )

        return MatterSimASECalculator(
            potential=self.potential,
            args_dict=self.args_dict,
            compute_stress=self.compute_stress,
            stress_weight=self.stress_weight,
            device=self.device,
        )
