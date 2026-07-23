"""Generates structures and tasks for coefficient of thermal expansion (CTE) workflows.

Prepares per-temperature structure/task inputs consumed by MD sampling to estimate the
volumetric CTE from temperature-dependent equilibrium volumes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from materialsframework.utils import to_structure

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class CTETransformation:
    """Build per-temperature structure/task inputs for CTE analysis."""

    def __init__(
        self,
        ensemble: str = "npt_berendsen",
        pressure: float = 1.0,
    ) -> None:
        """Initialize the transformation object.

        Args:
            ensemble (str, optional): Requested MD ensemble label. Defaults to "npt_berendsen".
            pressure (float, optional): Target pressure value in atm. Defaults to 1.0.
        """
        self.ensemble = ensemble
        self.pressure = pressure

    def apply_transformation(
        self,
        structure: Structure | Atoms,
        temperatures: Sequence[float],
        steps: int = 10000,
    ) -> dict[str, Any]:
        """Prepare structures and task metadata for each target temperature.

        Args:
            structure (Structure | Atoms): Input structure for MD sampling.
            temperatures (Sequence[float]): Target temperatures in Kelvin.
            steps (int, optional): Number of MD steps per temperature. Defaults to 10000.

        Raises:
            ValueError: If temperatures are invalid or steps is non-positive.

        Returns:
            Dictionary with keys:
                - ``structures``: Mapping of temperature (K) to a copy of the input structure.
                - ``tasks``: Per-temperature MD task metadata (temperature, steps, ensemble, pressure).
        """
        validated_temperatures = self._validate_temperatures(temperatures)
        if steps <= 0:
            raise ValueError("steps must be a positive integer.")

        structure = to_structure(structure)

        structures: dict[float, Structure] = {}
        tasks: list[dict[str, Any]] = []

        for temperature in validated_temperatures:
            temperature_value = float(temperature)
            structures[temperature_value] = structure.copy()
            tasks.append(
                {
                    "temperature": temperature_value,
                    "steps": int(steps),
                    "ensemble": self.ensemble,
                    "pressure": float(self.pressure),
                }
            )

        return {"structures": structures, "tasks": tasks}

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
