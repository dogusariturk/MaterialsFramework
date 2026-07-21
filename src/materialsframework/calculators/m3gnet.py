"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the M3GNet potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from torch import Tensor

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class M3GNetCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the M3GNet potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including
            "potential_energy", "forces", and "stress".

    References:
        - M3GNet: https://doi.org/10.1038/s43588-022-00349-3
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "M3GNet-PES-MatPES-r2SCAN-2025.2",
        state_attr: Tensor | None = None,
        stress_unit: Literal["eV/A3", "GPa"] = "GPa",
        stress_weight: float = 1.0,
        use_voigt: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the M3GNetCalculator with the specified model and calculation settings.

        Args:
            model (str, optional): The M3GNet model to use. Defaults to "M3GNet-PES-MatPES-r2SCAN-2025.2".
            state_attr (Tensor | None, optional): State attributes to include in the potential energy calculation.
                This allows for additional model customization. Defaults to None.
            stress_unit (Literal["eV/A3", "GPa"], optional): The unit for stress calculations. If set to "GPa", stress will be calculated in GPa.
            stress_weight (float, optional): Conversion factor from GPa to eV/ang^3. If set to 1.0, stress is
                calculated in GPa. Defaults to 1.0.
            use_voigt (bool, optional): Whether to use Voigt notation for stress. Defaults to False.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Examples:
            >>> m3gnet_calculator = M3GNetCalculator(model="M3GNet-MP-2021.2.8-PES")

        Note:
            The remaining parameters for the M3GNet potential are set to their default values.
        """
        super().__init__(**kwargs)

        # M3GNet specific attributes
        self.model = model
        self.state_attr = state_attr
        self.stress_unit = stress_unit
        self.stress_weight = stress_weight
        self.use_voigt = use_voigt

        self._calculator = None


    @lazy_property("_calculator")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the M3GNet potential, using `state_attr` and `stress_weight`.

        Returns:
            Calculator: The ASE Calculator object configured with the M3GNet potential.
        """
        import matgl
        from matgl.ext.ase import PESCalculator

        potential = matgl.load_model(path=self.model)

        return PESCalculator(
            potential=potential,
            state_attr=self.state_attr,
            stress_unit=self.stress_unit,
            stress_weight=self.stress_weight,
            use_voigt=self.use_voigt,
        )
