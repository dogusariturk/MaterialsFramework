from __future__ import annotations

from typing import TYPE_CHECKING

from ase import Atoms
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class SurfaceAnalyzer:
    """A class used to perform surface energy analysis for a given structure."""

    def __init__(
        self,
        miller_index: tuple[int, int, int] = (1, 1, 0),
        min_slab_size: float = 10.0,
        min_vacuum_size: float = 10.0,
        center_slab: bool = True,
        in_unit_planes: bool = False,
        primitive: bool = False,
        symmetrize: bool = True,
        calculator: BaseCalculator | None = None,
    ) -> None:
        """Initializes the `SurfaceAnalyzer` object.

        Args:
            miller_index (tuple[int, int, int], optional): The Miller index for the surface. Defaults to (1, 1, 0).
            min_slab_size (float, optional): The minimum slab size in Angstroms. Defaults to 10.0.
            min_vacuum_size (float, optional): The minimum vacuum size in Angstroms Defaults to 20.0.
            center_slab (bool, optional): Whether to center the slab. Defaults to True.
            in_unit_planes (bool, optional): Whether to use unit planes for slab generation. Defaults to True.
            primitive (bool, optional): Whether to use the primitive cell for slab generation. Defaults to False.
            symmetrize (bool, optional): Whether to symmetrize the slab. Defaults to True.
            calculator (BaseCalculator | None, optional): The calculator used for energy calculations. Defaults to `M3GNetCalculator`.
        """
        self.miller_index = miller_index
        self.min_slab_size = min_slab_size
        self.min_vacuum_size = min_vacuum_size
        self.center_slab = center_slab
        self.in_unit_planes = in_unit_planes
        self.primitive = primitive
        self.symmetrize = symmetrize

        self.ase_adaptor = AseAtomsAdaptor()
        self._calculator = calculator

    def calculate(
        self,
        structure: Structure | Atoms,
        is_relaxed: bool = False,
    ) -> dict[str, list | float]:
        """Calculates the surface energy of a given structure.

        Args:
            structure (Structure | Atoms): The undeformed structure to be analyzed.
            is_relaxed (bool, optional): Whether the structure is already relaxed. Defaults to False.

        Returns:
            dict[str, list | float]: A dictionary with the following keys:
                - `bulk_structure`: The bulk structure used for calculations.
                - `bulk_energy`: The total energy of the bulk structure.
                - `bulk_energy_per_atom`: The energy per atom of the bulk structure.
                - `slabs`: A list of dictionaries containing slab information with the following keys:
                    - `slab`: The slab structure.
                    - `relaxed_slab`: The relaxed slab structure.
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

        bulk_energy = self.calculator.calculate(structure)["energy"]
        bulk_energy_per_atom = bulk_energy / len(structure)

        slab_generator = SlabGenerator(
            initial_structure=structure,
            miller_index=self.miller_index,
            min_slab_size=self.min_slab_size,
            min_vacuum_size=self.min_vacuum_size,
            center_slab=self.center_slab,
            in_unit_planes=self.in_unit_planes,
            primitive=self.primitive,
        )

        slabs = slab_generator.get_slabs(symmetrize=self.symmetrize, repair=True)

        results = []
        for slab in slabs:
            slab_area = slab.surface_area
            res_slab = self.calculator.relax(slab)
            relaxed_slab = res_slab["final_structure"]
            slab_energy = res_slab["energy"]
            gamma = (slab_energy - len(relaxed_slab) * bulk_energy_per_atom) / (2 * slab_area)
            results.append(
                {
                    "slab": slab,
                    "relaxed_slab": relaxed_slab,
                    "slab_energy": slab_energy,
                    "slab_area": slab_area,
                    "gamma": gamma,
                }
            )

        return {
            "bulk_structure": structure,
            "bulk_energy": bulk_energy,
            "bulk_energy_per_atom": bulk_energy_per_atom,
            "slabs": results,
        }

    @property
    def calculator(self) -> BaseCalculator:
        """Returns the calculator instance used for energy calculations.

        If the calculator instance is not already initialized, this method creates a new `M3GNetCalculator` instance.

        Returns:
            BaseCalculator: The calculator object used for energy calculations.
        """
        if self._calculator is None:
            from materialsframework.calculators.m3gnet import M3GNetCalculator

            self._calculator = M3GNetCalculator()
        return self._calculator
