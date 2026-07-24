"""Calculator for computing the formation energy of materials with the MEGNet potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from materialsframework.utils import lazy_property, requires

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class MEGNetCalculator:
    """Calculator for the formation energy of a structure using the MEGNet potential, via the `matgl` package.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute. Currently, only
            "formation_energy" is supported.

    References:
        - MEGNet: https://doi.org/10.1021/acs.chemmater.9b01294
    """

    AVAILABLE_PROPERTIES: list[str] = ["formation_energy"]

    def __init__(
        self,
        model: str = "MEGNet-Eform-MP-2018.6.1",
    ) -> None:
        """Initializes a MEGNetCalculator instance with the specified MEGNet model.

        Args:
            model (str, optional): The name of the MEGNet model to use for calculations. Defaults to
                "MEGNet-Eform-MP-2018.6.1".

        Examples:
            >>> megnet_calculator = MEGNetCalculator(model="MEGNet-Eform-MP-2018.6.1")

        Note:
            The remaining parameters for the MEGNet potential are set to their default values.
        """
        # MEGNet specific attributes
        self.model = model

        self._potential = None

    @lazy_property("_potential")
    @requires("matgl", extra="matgl")
    def potential(self) -> Any:
        """Lazily loads and caches the MEGNet potential specified during initialization.

        Returns:
            Any: The loaded `matgl` MEGNet potential.
        """
        import matgl

        return matgl.load_model(self.model)

    def calculate(self, structure: Structure) -> dict[str, float]:
        """Calculates the formation energy of the provided structure using the MEGNet model.

        Args:
            structure (Structure): A Pymatgen `Structure` object representing the material structure for which the
                formation energy will be calculated.

        Returns:
            dict[str, float]: A dictionary containing the formation energy of the structure under the key "formation_energy".

        Examples:
            >>> struct = Structure.from_file("POSCAR")
            >>> megnet_calculator = MEGNetCalculator()
            >>> result = megnet_calculator.calculate(structure=struct)
        """
        return {"formation_energy": float(self.potential.predict_structure(structure))}
