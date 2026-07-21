"""Calculator for computing the formation energy of materials with the MEGNet potential."""

from __future__ import annotations

from typing import TYPE_CHECKING

from materialsframework.utils import lazy_property

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
        model: str = "MEGNet-MP-2018.6.1-Eform",
    ) -> None:
        """Initializes a MEGNetCalculator instance with the specified MEGNet model.

        Args:
            model (str, optional): The name of the MEGNet model to use for calculations. Defaults to
                "MEGNet-MP-2018.6.1-Eform".

        Examples:
            >>> megnet_calculator = MEGNetCalculator(model="MEGNet-MP-2018.6.1-Eform")

        Note:
            The remaining parameters for the MEGNet potential are set to their default values.
        """
        # MEGNet specific attributes
        self.model = model

        self._potential = None

    @lazy_property("_potential")
    def potential(self):
        """Lazily loads and returns the MEGNet potential specified during initialization.

        Returns:
            MEGNet: The loaded MEGNet model instance used for calculations.
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
        return {"formation_energy": self.potential.predict_structure(structure)}
