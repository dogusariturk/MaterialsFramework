"""This module provides a class to perform a Bain transformation on a given structure.

The `BainPathAnalyzer` class calculates the potential energies along the Bain transformation path,
which describes the structural transition between body-centered cubic (BCC) and face-centered cubic
(FCC) phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from materialsframework.analysis.base import BaseAnalyzer
from materialsframework.analysis.utils import require_properties
from materialsframework.transformations.bain import BainDisplacementTransformation
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class BainPathAnalyzer(BaseAnalyzer):
    """A class used to analyze the Bain transformation path for a given structure.

    Applies a Bain transformation to an undeformed structure and calculates the potential energy at
    each c/a ratio along the transformation path.
    """

    def __init__(
        self,
        start: float = 0.89,
        stop: float = 1.5,
        step: float = 0.01,
        calculator: BaseCalculator | None = None,
        bain_transformation: BainDisplacementTransformation | None = None,
    ) -> None:
        """Initializes the `BainPathAnalyzer` object.

        Args:
            start (float, optional): The starting displacement value for the c/a ratio. Defaults to 0.89.
            stop (float, optional): The stopping displacement value for the c/a ratio. Defaults to 1.5.
            step (float, optional): The step size for incrementing the c/a ratio. Defaults to 0.01.
            calculator (BaseCalculator | None, optional): The calculator object used to compute potential energies.
            bain_transformation (BainDisplacementTransformation | None, optional): The transformation object used to
                apply Bain displacements. If not provided, a new instance is initialized.
        """
        super().__init__(calculator)
        self.start = start
        self.stop = stop
        self.step = step
        self._bain_transformation = bain_transformation

    @require_properties("energy")
    def calculate(self, structure: Structure | Atoms, is_relaxed: bool = False) -> dict[str, list]:
        """Calculates the potential energies along the Bain Path for the given undeformed structure.

        Applies the Bain transformation to the input structure to generate deformed structures at a
        series of c/a ratios, then evaluates the potential energy of each with the provided calculator.

        Args:
            structure (Structure | Atoms): The undeformed structure to be transformed and analyzed.
            is_relaxed (bool, optional): Whether the input structure is already relaxed. Defaults to False.

        Returns:
            dict[str, list]: Dictionary with keys:
                - ``c_a_list``: c/a ratios for each deformed structure.
                - ``energy_list``: Potential energy for each deformed structure.

        Raises:
            ValueError: If the calculator object does not have the 'energy' property implemented.
        """
        structure = self._ensure_relaxed(structure, is_relaxed)

        displaced_structures = self.bain_transformation.apply_transformation(structure=structure)

        c_a_list, energy_list = zip(
            *[
                (c_a, self.calculator.calculate(structure=deformed_structure)["energy"])
                for c_a, deformed_structure in displaced_structures.items()
            ],
            strict=False,
        )

        return {"c_a_list": c_a_list, "energy_list": energy_list}

    @lazy_property("_bain_transformation")
    def bain_transformation(self) -> BainDisplacementTransformation:
        """Returns the Bain displacement transformation object used to apply Bain displacements.

        Returns:
            BainDisplacementTransformation: The transformation object used for Bain displacements.
        """
        return BainDisplacementTransformation(start=self.start, stop=self.stop, step=self.step)
