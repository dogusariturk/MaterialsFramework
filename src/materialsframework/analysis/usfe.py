"""This module provides a class to perform a USFE analysis on a given structure.

The `USFEAnalyzer` class computes a generalized stacking fault energy (GSFE) curve by
combining rigid displacement structures from `USFETransformation` with a calculator.
The unstable stacking fault energy (USFE) is extracted as the maximum GSFE value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from materialsframework.analysis.base import BaseAnalyzer
from materialsframework.analysis.utils import require_properties
from materialsframework.constants import EV_A2_TO_MJ_M2
from materialsframework.transformations.usfe import USFETransformation
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class USFEAnalyzer(BaseAnalyzer):
    """A class used to compute GSFE curves and USFE values for BCC-like slip systems."""

    def __init__(
        self,
        slip_plane: str = "110",
        start: float = 0.0,
        stop: float = 1.0,
        num_steps: int = 11,
        calculator: BaseCalculator | None = None,
        usfe_transformation: USFETransformation | None = None,
    ) -> None:
        """Initializes the `USFEAnalyzer` object.

        Args:
            slip_plane (str, optional): Slip plane label, either ``"110"`` or ``"112"``.
                Defaults to ``"110"``.
            start (float, optional): Starting displacement fraction. Defaults to 0.0.
            stop (float, optional): Ending displacement fraction. Defaults to 1.0.
            num_steps (int, optional): Number of displacement points. Defaults to 11.
            calculator (BaseCalculator | None, optional): Calculator for energy evaluations.
            usfe_transformation (USFETransformation | None, optional): Custom transformation
                object. If not provided, a default one is created lazily.

        Raises:
            ValueError: If ``slip_plane`` is not supported.
        """
        if slip_plane not in {"110", "112"}:
            raise ValueError(f"Unsupported slip_plane '{slip_plane}'. Supported values are: 110, 112.")

        super().__init__(calculator)
        self.slip_plane = slip_plane
        self.start = start
        self.stop = stop
        self.num_steps = num_steps

        self._usfe_transformation = usfe_transformation

    @require_properties("energy")
    def calculate(
        self, structure: Structure | Atoms, is_relaxed: bool = False
    ) -> dict[str, str | int | float | list[dict[str, float]]]:
        """Calculates GSFE curve and USFE for a given structure.

        Args:
            structure (Structure | Atoms): Input structure.
            is_relaxed (bool, optional): Whether input is already relaxed. Defaults to False.

        Returns:
            dict[str, str | int | float | list[dict[str, float]]]: Dictionary with keys:
                - ``gsfe_curve``: GSFE samples for each displacement step.
                - ``usfe_mJ_m2``: Maximum GSFE value (unstable stacking fault energy) in mJ/m^2.
                - ``usfe_displacement_frac``: Displacement fraction at the USFE peak.
                - ``slip_plane``: Slip plane used for displacement.
                - ``num_steps``: Number of displacement steps used in the scan.

        Raises:
            ValueError: If the calculator does not provide ``energy``.
        """
        structure = self._ensure_relaxed(structure, is_relaxed)

        transformation_result = self.usfe_transformation.apply_transformation(structure)
        displaced_structures = transformation_result["displaced_structures"]
        fault_area = transformation_result["fault_area"]

        energy_points = [
            (
                frac,
                self.calculator.calculate(displaced_structure)["energy"],
            )
            for frac, displaced_structure in displaced_structures.items()
        ]

        reference_energy = energy_points[0][1]
        if fault_area is None or fault_area <= 0:
            raise ValueError("Fault area must be a positive value after transformation.")

        gsfe_curve = [
            {
                "displacement_fraction": float(frac),
                "gamma_mJ_m2": float((energy - reference_energy) / fault_area * EV_A2_TO_MJ_M2),
            }
            for frac, energy in energy_points
        ]

        usfe_point = max(gsfe_curve, key=lambda point: point["gamma_mJ_m2"])

        return {
            "gsfe_curve": gsfe_curve,
            "usfe_mJ_m2": usfe_point["gamma_mJ_m2"],
            "usfe_displacement_frac": usfe_point["displacement_fraction"],
            "slip_plane": self.slip_plane,
            "num_steps": len(gsfe_curve),
        }

    @lazy_property("_usfe_transformation")
    def usfe_transformation(self) -> USFETransformation:
        """Return USFE transformation object for displaced structures.

        Returns:
            USFE transformation instance.
        """
        return USFETransformation(
            slip_plane=self.slip_plane,
            start=self.start,
            stop=self.stop,
            num_steps=self.num_steps,
        )
