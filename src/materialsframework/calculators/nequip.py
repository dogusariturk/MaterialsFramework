"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the NequIP potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class NequIPCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the NequIP potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stresses".

    References:
        - NequIP: https://arxiv.org/abs/2504.16068
    """

    AVAILABLE_PROPERTIES = ["energy", "energies", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "",
        device: Literal["cpu", "cuda"] = "cpu",
        chemical_species_to_atom_type_map: dict[str, str] | bool | None = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the NequIPCalculator with the specified model and calculation settings.

        Args:
            model (str, optional): The NequIP model to use. Defaults to "".
            device (Literal["cuda", "cpu"], optional): The device to use for calculations. Defaults to "cpu".
            chemical_species_to_atom_type_map (Optional[Union[Dict[str, str], bool]], optional): A mapping from chemical species to atom
                types expected by the NequIP model. Defaults to True, which means that the mapping will be automatically inferred from the
                model.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        # NequIP specific attributes
        self.model = model
        self.device = device
        self.chemical_species_to_atom_type_map = chemical_species_to_atom_type_map

        self._calculator = None

    @lazy_property("_calculator")
    @requires("nequip", extra="nequip")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the NequIP potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the NequIP potential.
        """
        from nequip.integrations.ase import NequIPCalculator

        return NequIPCalculator.from_compiled_model(
            compile_path=self.model,
            device=self.device,
            chemical_species_to_atom_type_map=self.chemical_species_to_atom_type_map,
        )
