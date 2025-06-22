from __future__ import annotations

from typing import TYPE_CHECKING

from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import SlabGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class SurfaceAnalyzer:
    """
    A class used to perform surface energy analysis for a given structure.
    """
    def __init__(
            self,
            calculator: BaseCalculator | None = None
    ) -> None:
        """
        Initializes the `SurfaceAnalyzer` object.

        Args:
            calculator (BaseCalculator | None, optional): The calculator used for energy calculations. Defaults to `M3GNetCalculator`.
        """
        self.ase_adaptor = AseAtomsAdaptor()
        self._calculator = calculator

    def calculate(
            self,
            structure: Structure | Atoms,
            is_relaxed: bool = False,
            miller_index: tuple[int, int, int] = (1, 1, 1),
            min_slab_size: float = 10.0,
            min_vacuum_size: float = 10.0
    ) -> dict[str, list | float]:
        """
        Calculates the surface energy of a given structure.

        Args:
            structure (Structure | Atoms): The undeformed structure to be analyzed.
            is_relaxed (bool, optional): Whether the structure is already relaxed. Defaults to False.
            miller_index (tuple[int, int, int], optional): The Miller index for the surface. Defaults to (1, 1, 1).
            min_slab_size (float, optional): The minimum slab size in Angstroms. Defaults to 10.0.
            min_vacuum_size (float, optional): The minimum vacuum size in Angstroms. Defaults to 10.0.

        Returns:
            dict[str, list | float]: A dictionary with the following keys:
                - `bulk_structure`: The bulk structure used for calculations.
                - `bulk_energy`: The energy per atom of the bulk structure.
                - `slabs`: A list of dictionaries containing slab information:
                    - `slab`: The slab structure.
                    - `slab_energy`: The energy of the slab.
                    - `slab_area`: The surface area of the slab.
                    - `gamma`: The surface energy of the slab.
        """
        if "energy" not in self.calculator.AVAILABLE_PROPERTIES:
            raise ValueError("The calculator object must have the 'energy' property implemented.")

        if isinstance(structure, Atoms):
            structure = self.ase_adaptor.get_structure(structure)

        if not is_relaxed:
            self.calculator.relax_cell = True
            structure: Structure = self.calculator.relax(structure)["final_structure"]
            self.calculator.relax_cell = False

        slab_generator = SlabGenerator(
                initial_structure=structure,
                miller_index=miller_index,
                min_slab_size=min_slab_size,
                min_vacuum_size=min_vacuum_size
        )

        slabs = slab_generator.get_slabs()

        return {
            "bulk_structure": structure,
            "bulk_energy": self.calculator.calculate(structure)['energy'] / len(structure),
            "slabs": [
                {
                    "slab": slab,
                    "slab_energy": self.calculator.calculate(slab)["energy"],
                    "slab_area": slab.surface_area,
                    "gamma": (
                        self.calculator.calculate(slab)["energy"]
                        - len(slab) * (self.calculator.calculate(structure)['energy'] / len(structure))
                    ) / (2 * slab.surface_area)
                }
                for slab in slabs
            ]
        }

    @property
    def calculator(self) -> BaseCalculator:
        """
        Returns the calculator instance used for energy calculations.

        If the calculator instance is not already initialized, this method creates a new `M3GNetCalculator` instance.

        Returns:
            BaseCalculator: The calculator object used for energy calculations.
        """
        if self._calculator is None:
            from materialsframework.calculators.m3gnet import M3GNetCalculator
            self._calculator = M3GNetCalculator()
        return self._calculator
