"""This module provides a class to perform a coefficient of thermal expansion (CTE) analysis on a given structure.

The `CTEAnalyzer` class estimates the volumetric coefficient of thermal expansion by running
temperature-dependent MD sampling through `CTETransformation` and fitting the resulting
volume-temperature data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from materialsframework.analysis.base import BaseAnalyzer
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.transformations.cte import CTETransformation
from materialsframework.utils import lazy_property, to_structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from pymatgen.core import Structure

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class CTEAnalyzer(BaseAnalyzer):
    """A class used to estimate volumetric CTE from NPT-MD volume-temperature data."""

    def __init__(
        self,
        temperatures: Sequence[float] = (300.0, 600.0, 900.0),
        ensemble: str = "npt_berendsen",
        pressure: float = 1.0,
        calculator: BaseCalculator | None = None,
        cte_transformation: CTETransformation | None = None,
    ) -> None:
        """Initializes the `CTEAnalyzer` object.

        Args:
            temperatures (Sequence[float], optional): Temperatures in Kelvin to sample with MD.
                Defaults to ``(300.0, 600.0, 900.0)``.
            ensemble (str, optional): MD ensemble applied to the calculator for every temperature in the
                sweep; must be one of the ensembles supported by `BaseMDCalculator`. Defaults to
                "npt_berendsen".
            pressure (float, optional): Target pressure in atm, applied to the calculator for every
                temperature in the sweep. Defaults to 1.0.
            calculator (BaseCalculator | None, optional): MD-capable calculator.
            cte_transformation (CTETransformation | None, optional): Transformation object used to
                generate the per-temperature structure copies. Defaults to a `CTETransformation` built
                from `temperatures`.
        """
        super().__init__(calculator)
        self.temperatures = temperatures
        self.ensemble = ensemble
        self.pressure = pressure
        self._cte_transformation = cte_transformation

    def calculate(self, structure: Structure | Atoms, steps: int = 10000) -> dict[str, list | float]:
        """Calculates temperature-dependent volumes and volumetric CTE.

        Args:
            structure (Structure | Atoms): Input structure.
            steps (int, optional): Number of MD steps per temperature. Defaults to ``10000``.

        Raises:
            ValueError: If `steps` is non-positive, fewer than two distinct temperatures are
                configured, or the calculator is not a `BaseMDCalculator`.

        Returns:
            dict[str, list | float]: Dictionary with keys:
                - ``temperatures``: Configured temperatures in Kelvin.
                - ``volumes``: Final volume for each temperature in A^3.
                - ``cte``: Volumetric CTE in K^-1.
                - ``cte_ppm``: Volumetric CTE in ppm/K.
        """
        if steps <= 0:
            raise ValueError("steps must be a positive integer.")

        structure = to_structure(structure)

        calculator = self.calculator
        if not isinstance(calculator, BaseMDCalculator):
            raise ValueError("The calculator object must be a `BaseMDCalculator` to run MD sampling for CTE.")

        cte_structures = self.cte_transformation.apply_transformation(structure)
        if len(cte_structures) < 2:
            raise ValueError("At least two distinct temperatures are required to compute CTE.")

        prev_md_state = calculator.ensemble, calculator.pressure, calculator.temperature
        calculator.ensemble = self.ensemble
        calculator.pressure = self.pressure

        try:
            temperatures: list[float] = []
            volumes: list[float] = []
            for temperature, temperature_structure in cte_structures.items():
                calculator.temperature = temperature
                md_result = calculator.run(structure=temperature_structure, steps=steps)
                temperatures.append(temperature)
                volumes.append(float(md_result["final_structure"].volume))
        finally:
            calculator.ensemble, calculator.pressure, calculator.temperature = prev_md_state

        slope, _ = np.polyfit(temperatures, volumes, 1)
        reference_volume = volumes[int(np.argmin(temperatures))]

        alpha_volumetric = float(slope / reference_volume)

        return {
            "temperatures": temperatures,
            "volumes": volumes,
            "cte": alpha_volumetric,
            "cte_ppm": alpha_volumetric * 1e6,
        }

    @lazy_property("_cte_transformation")
    def cte_transformation(self) -> CTETransformation:
        """Return CTE transformation object used by this analyzer.

        Returns:
            CTE transformation instance.
        """
        return CTETransformation(temperatures=self.temperatures)
