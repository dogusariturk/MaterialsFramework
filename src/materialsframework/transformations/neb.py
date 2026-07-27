"""Generates interpolated images for NEB calculations.

Produces a series of intermediate structures between two endpoint structures, for use as
the initial reaction path in a Nudged Elastic Band (NEB) calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class NEBTransformation:
    """Generates interpolated images between two endpoint structures for NEB calculations."""

    def __init__(
        self,
        n_images: int = 5,
        interpolate_lattices: bool = False,
        pbc: bool = True,
        autosort_tol: float = 0.5,
        end_amplitude: float = 1,
    ) -> None:
        """Initializes the `NEBTransformation` object.

        Args:
            n_images (int, optional): Number of intermediate images to interpolate between the two
                endpoint structures. Defaults to 5.
            interpolate_lattices (bool, optional): Whether to interpolate lattices between images.
                Defaults to False.
            pbc (bool, optional): Whether to use periodic boundary conditions to find the shortest
                path between endpoints. Defaults to True.
            autosort_tol (float, optional): Distance tolerance (in Angstrom) used to automatically
                match sites between the initial and final structures. A value of 0.5 usually works
                well for NEB calculations; 0 disables sorting. Defaults to 0.5.
            end_amplitude (float, optional): Fractional amplitude of the endpoint of the
                interpolation. A value of 1 corresponds to full distortion to `final_structure`.
                Defaults to 1.
        """
        self.n_images = n_images
        self.interpolate_lattices = interpolate_lattices
        self.pbc = pbc
        self.autosort_tol = autosort_tol
        self.end_amplitude = end_amplitude

    def apply_transformation(self, initial_structure: Structure, final_structure: Structure) -> list[Structure]:
        """Generates interpolated images between the initial and final structures.

        Args:
            initial_structure (Structure): The initial (relaxed) endpoint structure.
            final_structure (Structure): The final (relaxed) endpoint structure.

        Returns:
            list[Structure]: The interpolated images, including `initial_structure` and `final_structure`
                as the first and last elements, respectively.
        """
        return initial_structure.interpolate(
            end_structure=final_structure,
            nimages=self.n_images,
            interpolate_lattices=self.interpolate_lattices,
            pbc=self.pbc,
            autosort_tol=self.autosort_tol,
            end_amplitude=self.end_amplitude,
        )
