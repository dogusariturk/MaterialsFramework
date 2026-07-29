"""Generates surface slabs for surface energy calculations.

Produces a set of terminated, vacuum-padded slab structures for a given Miller index from a bulk
structure, for use in `SurfaceAnalyzer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymatgen.core.surface import SlabGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.core.surface import Slab

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class SurfaceTransformation:
    """Generates surface slabs from a bulk structure for surface energy calculations."""

    def __init__(
        self,
        miller_index: tuple[int, int, int] = (1, 1, 0),
        min_slab_size: float = 10.0,
        min_vacuum_size: float = 10.0,
        center_slab: bool = True,
        in_unit_planes: bool = False,
        primitive: bool = False,
        symmetrize: bool = True,
    ) -> None:
        """Initializes the `SurfaceTransformation` object.

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
        """
        self.miller_index = miller_index
        self.min_slab_size = min_slab_size
        self.min_vacuum_size = min_vacuum_size
        self.center_slab = center_slab
        self.in_unit_planes = in_unit_planes
        self.primitive = primitive
        self.symmetrize = symmetrize

    def apply_transformation(self, structure: Structure) -> list[Slab]:
        """Generates surface slabs for the configured Miller index from a bulk structure.

        Args:
            structure (Structure): The (relaxed) bulk structure used to generate slabs.

        Returns:
            list[Slab]: The generated slabs, repaired to remove broken bonds at the surface and,
                if ``symmetrize`` is True, restricted to terminations with symmetric surfaces.
        """
        slab_generator = SlabGenerator(
            initial_structure=structure,
            miller_index=self.miller_index,
            min_slab_size=self.min_slab_size,
            min_vacuum_size=self.min_vacuum_size,
            center_slab=self.center_slab,
            in_unit_planes=self.in_unit_planes,
            primitive=self.primitive,
        )
        return slab_generator.get_slabs(symmetrize=self.symmetrize, repair=True)
