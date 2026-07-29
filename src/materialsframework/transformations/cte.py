"""Generates per-temperature structure copies for coefficient of thermal expansion (CTE) workflows.

Prepares the per-temperature structure inputs consumed by MD sampling to estimate the volumetric
CTE from temperature-dependent equilibrium volumes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from materialsframework.utils import to_structure

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class CTETransformation:
    """Copies the input structure once per requested temperature for CTE analysis."""

    def __init__(self, temperatures: Sequence[float]) -> None:
        """Initializes the transformation object.

        Args:
            temperatures (Sequence[float]): Target temperatures in Kelvin.

        Raises:
            ValueError: If temperatures are empty, non-numeric, non-finite, or non-positive.
        """
        self._temperatures = self._validate_temperatures(temperatures)

    def apply_transformation(self, structure: Structure | Atoms) -> dict[float, Structure]:
        """Generate one structure copy per target temperature.

        Args:
            structure (Structure | Atoms): Input structure for MD sampling.

        Returns:
            dict[float, Structure]: Mapping of temperature (K) to a copy of the input structure.
        """
        structure = to_structure(structure)
        return {temperature: structure.copy() for temperature in self._temperatures}

    @staticmethod
    def _validate_temperatures(temperatures: Sequence[float]) -> list[float]:
        """Validate and return temperatures as floats.

        Args:
            temperatures: Candidate temperatures in Kelvin.

        Raises:
            ValueError: If temperatures are empty, non-numeric, non-finite, or non-positive.

        Returns:
            Validated temperatures.
        """
        if not isinstance(temperatures, Sequence) or isinstance(temperatures, str):
            raise ValueError("temperatures must be provided as a non-empty sequence of positive values in Kelvin.")
        if len(temperatures) == 0:
            raise ValueError("temperatures must contain at least one value.")

        validated_temperatures: list[float] = []
        for temperature in temperatures:
            if not isinstance(temperature, int | float):
                raise ValueError("All temperatures must be numeric values in Kelvin.")
            temperature_value = float(temperature)
            if not np.isfinite(temperature_value):
                raise ValueError("All temperatures must be finite values in Kelvin.")
            if temperature_value <= 0:
                raise ValueError("All temperatures must be greater than 0 K.")
            validated_temperatures.append(temperature_value)
        return validated_temperatures
