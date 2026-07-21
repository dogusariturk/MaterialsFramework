"""Calculator for computing potential energy, forces, and stresses, and for relaxing structures, with the Allegro potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class AllegroCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the Allegro potential.

    Allegro is a strictly local equivariant model architecture distributed as a plugin for the `nequip`
    training and inference framework. Trained models are compiled and loaded through the same ASE interface
    as `NequIPCalculator`; installing the `nequip-allegro` package registers the Allegro model builders with
    `nequip` so that compiled Allegro models can be deserialized correctly.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", and "stress".

    References:
        - Allegro: https://doi.org/10.1038/s41467-023-36329-y
    """

    AVAILABLE_PROPERTIES = ["energy", "energies", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "",
        device: Literal["cpu", "cuda"] = "cpu",
        chemical_species_to_atom_type_map: dict[str, str] | bool | None = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the AllegroCalculator with the specified model and calculation settings.

        Args:
            model (str): The path to a compiled Allegro model.
            device (Literal["cuda", "cpu"]): The device to use for calculations. Defaults to "cpu".
            chemical_species_to_atom_type_map (Optional[Union[Dict[str, str], bool]], optional): A mapping from chemical species to atom
                types expected by the Allegro model. Defaults to True, which means that the mapping will be automatically inferred from
                the model.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.
        """
        super().__init__(**kwargs)

        # Allegro specific attributes
        self.model = model
        self.device = device
        self.chemical_species_to_atom_type_map = chemical_species_to_atom_type_map

        self._calculator = None

    @lazy_property("_calculator")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the Allegro potential, using the settings from initialization.

        Returns:
            Calculator: The ASE Calculator object configured with the Allegro potential.
        """
        import allegro  # noqa: F401
        from nequip.integrations.ase import NequIPCalculator

        return NequIPCalculator.from_compiled_model(
            compile_path=self.model,
            device=self.device,
            chemical_species_to_atom_type_map=self.chemical_species_to_atom_type_map,
        )
