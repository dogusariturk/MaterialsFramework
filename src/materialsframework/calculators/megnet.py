"""
This module provides a class for performing calculations using the MEGNet potential.

The `MEGNetCalculator` class is designed to calculate properties such as the formation energy
of materials using a specified MEGNet model.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class MEGNetCalculator:
    """
    A calculator class for performing material property calculations using the MEGNet potential.

    The `MEGNetCalculator` class is capable of calculating the formation energy of a given structure
    using a specified MEGNet model. The class is designed to interface with the `matgl` package,
    leveraging its models to make predictions.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute.
                                          Currently, only "formation_energy" is supported.
    """
    AVAILABLE_PROPERTIES: list[str] = ["formation_energy"]

    def __init__(
            self,
            model: str = "MEGNet-MP-2018.6.1-Eform",
    ) -> None:
        """
        Initializes a MEGNetCalculator instance with the specified MEGNet model.

        This method sets up the calculator with a predefined MEGNet model, which will be used
        for predicting material properties such as formation energy.

        Args:
            model (str, optional): The name of the MEGNet model to use for calculations.
                                   Defaults to "MEGNet-MP-2018.6.1-Eform".

        Examples:
            >>> megnet_calculator = MEGNetCalculator(model="MEGNet-MP-2018.6.1-Eform")

        Note:
            The remaining parameters for the MEGNet potential are set to their default values.
        """
        import matgl
        from matgl.models import MEGNet

        # MEGNet specific attributes
        self.model = model

        self._calculator = None
        self._potential = None

    @property
    def potential(self) -> MEGNet:
        """
        Loads and returns the MEGNet potential associated with this calculator instance.

        This property lazily loads the MEGNet model specified during initialization if it
        has not already been loaded. The loaded potential is then used for all subsequent
        calculations.

        Returns:
            MEGNet: The loaded MEGNet model instance used for calculations.
        """
        if self._potential is None:
            self._potential = matgl.load_model(self.model)
        return self._potential

    def calculate(
            self,
            structure: Structure
    ) -> dict[str, float]:
        """
        Calculates the formation energy of the provided structure using the MEGNet model.

        This method predicts the formation energy for a given structure using the preloaded
        MEGNet model.

        Args:
            structure (Structure): A Pymatgen `Structure` object representing the material structure
                                   for which the formation energy will be calculated.

        Returns:
            dict[str, float]: A dictionary containing the formation energy of the structure under the key
                              "formation_energy".

        Examples:
            >>> struct = Structure.from_file("POSCAR")
            >>> megnet_calculator = MEGNetCalculator()
            >>> result = megnet_calculator.calculate(structure=struct)
        """
        return {
                "formation_energy": self.potential.predict_structure(structure)
        }
