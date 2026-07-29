"""This module provides a class to perform surface energy analysis on a given structure.

The `SurfaceAnalyzer` class computes the surface energy of one or more slab terminations by
combining slab structures from `SurfaceTransformation` with a calculator. The bulk structure is
relaxed (cell and atoms) to obtain a reference energy per atom, then each slab is relaxed with the
cell fixed (only atomic positions relax) so the vacuum spacing is preserved, and the surface energy
is extracted from the difference between the relaxed slab energy and the equivalent bulk energy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from materialsframework.analysis.base import BaseAnalyzer
from materialsframework.analysis.utils import require_properties
from materialsframework.constants import EV_A2_TO_J_M2
from materialsframework.transformations.surface import SurfaceTransformation
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class SurfaceAnalyzer(BaseAnalyzer):
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
        surface_transformation: SurfaceTransformation | None = None,
    ) -> None:
        """Initializes the `SurfaceAnalyzer` object.

        Args:
            miller_index (tuple[int, int, int], optional): The Miller index for the surface. Defaults to (1, 1, 0).
            min_slab_size (float, optional): The minimum slab size in Angstroms. Defaults to 10.0.
            min_vacuum_size (float, optional): The minimum vacuum size in Angstroms. Defaults to 10.0.
            center_slab (bool, optional): Whether to center the slab within the vacuum. Defaults to True.
            in_unit_planes (bool, optional): Whether ``min_slab_size`` and ``min_vacuum_size`` are in units of
                hkl planes instead of Angstroms. Defaults to False.
            primitive (bool, optional): Whether to reduce the generated slabs to their primitive cell. Defaults to False.
            symmetrize (bool, optional): Whether to generate slabs with symmetric surface terminations on both
                sides. Defaults to True.
            calculator (BaseCalculator | None, optional): The calculator used for energy calculations. Defaults to
                a lazily constructed default calculator.
            surface_transformation (SurfaceTransformation | None, optional): The transformation object used to
                generate the slabs. If not provided, a new instance is initialized from `miller_index`,
                `min_slab_size`, `min_vacuum_size`, `center_slab`, `in_unit_planes`, `primitive`, and `symmetrize`.
        """
        super().__init__(calculator)
        self.miller_index = miller_index
        self.min_slab_size = min_slab_size
        self.min_vacuum_size = min_vacuum_size
        self.center_slab = center_slab
        self.in_unit_planes = in_unit_planes
        self.primitive = primitive
        self.symmetrize = symmetrize

        self._surface_transformation = surface_transformation

    @require_properties("energy")
    def calculate(
        self,
        structure: Structure | Atoms,
        is_relaxed: bool = False,
    ) -> dict[str, Structure | float | list[dict[str, Structure | float]]]:
        """Calculates the surface energy of a given structure for the configured Miller index.

        Args:
            structure (Structure | Atoms): The bulk structure to be analyzed.
            is_relaxed (bool, optional): Whether the structure is already relaxed. Defaults to False.

        Returns:
            dict[str, Structure | float | list[dict[str, Structure | float]]]: A dictionary with the following keys:
                - ``bulk_structure``: The (relaxed) bulk structure used as the energy reference.
                - ``bulk_energy``: The total energy of the bulk structure.
                - ``bulk_energy_per_atom``: The energy per atom of the bulk structure.
                - ``slabs``: A list of dictionaries, one per generated slab termination, with keys:
                    - ``slab``: The unrelaxed slab structure.
                    - ``relaxed_slab``: The slab structure after relaxation with the cell fixed.
                    - ``slab_energy``: The energy of the relaxed slab.
                    - ``slab_area``: The surface area of the slab, in Angstrom squared.
                    - ``gamma``: The surface energy of the slab, in eV/Angstrom squared.
                    - ``gamma_J_m2``: The surface energy of the slab, in J/m^2.

        Raises:
            ValueError: If the calculator does not implement the 'energy' property.
        """
        structure = self._ensure_relaxed(structure, is_relaxed)

        bulk_energy = self.calculator.calculate(structure)["energy"]
        bulk_energy_per_atom = bulk_energy / structure.num_sites

        slabs = self.surface_transformation.apply_transformation(structure)

        prev_relax_cell = self.calculator.relax_cell
        self.calculator.relax_cell = False
        try:
            results = []
            for slab in slabs:
                slab_area = slab.surface_area
                relaxed = self.calculator.relax(slab)
                relaxed_slab = relaxed["final_structure"]
                slab_energy = relaxed["energy"]
                gamma = (slab_energy - relaxed_slab.num_sites * bulk_energy_per_atom) / (2 * slab_area)
                results.append(
                    {
                        "slab": slab,
                        "relaxed_slab": relaxed_slab,
                        "slab_energy": slab_energy,
                        "slab_area": slab_area,
                        "gamma": gamma,
                        "gamma_J_m2": gamma * EV_A2_TO_J_M2,
                    }
                )
        finally:
            self.calculator.relax_cell = prev_relax_cell

        return {
            "bulk_structure": structure,
            "bulk_energy": bulk_energy,
            "bulk_energy_per_atom": bulk_energy_per_atom,
            "slabs": results,
        }

    @lazy_property("_surface_transformation")
    def surface_transformation(self) -> SurfaceTransformation:
        """Returns the transformation object used to generate the slabs.

        Returns:
            SurfaceTransformation: The transformation object used to generate the slabs.
        """
        return SurfaceTransformation(
            miller_index=self.miller_index,
            min_slab_size=self.min_slab_size,
            min_vacuum_size=self.min_vacuum_size,
            center_slab=self.center_slab,
            in_unit_planes=self.in_unit_planes,
            primitive=self.primitive,
            symmetrize=self.symmetrize,
        )
