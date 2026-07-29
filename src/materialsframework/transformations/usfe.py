"""Generates structures for USFE calculations.

Creates rigidly displaced structures along a selected BCC slip system, for generalized
stacking fault energy (GSFE) and unstable stacking fault energy (USFE) analyses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class USFETransformation:
    """Generates rigidly displaced structures for USFE calculations.

    Supports BCC-like slip planes ``110`` and ``112``. For each displacement fraction, atoms in
    the upper half-space relative to the selected plane are shifted along the corresponding
    in-plane slip direction.
    """

    _SLIP_SYSTEMS = {
        "110": {
            "plane": np.array([1.0, 1.0, 0.0]),
            "direction": np.array([1.0, -1.0, 1.0]),
        },
        "112": {
            "plane": np.array([1.0, 1.0, 2.0]),
            "direction": np.array([1.0, 1.0, -1.0]),
        },
    }

    def __init__(
        self,
        slip_plane: str = "110",
        start: float = 0.0,
        stop: float = 1.0,
        num_steps: int = 11,
    ) -> None:
        """Initializes the `USFETransformation` object.

        Args:
            slip_plane (str, optional): Slip plane label, either ``"110"`` or ``"112"``.
                Defaults to ``"110"``.
            start (float, optional): Starting displacement fraction. Defaults to 0.0.
            stop (float, optional): Ending displacement fraction. Defaults to 1.0.
            num_steps (int, optional): Number of displacement points. Defaults to 11.

        Raises:
            ValueError: If ``slip_plane`` is not supported.
        """
        if slip_plane not in self._SLIP_SYSTEMS:
            raise ValueError(f"Unsupported slip_plane '{slip_plane}'. Supported values are: 110, 112.")

        self.slip_plane = slip_plane
        self.displacement_fractions = np.linspace(start, stop, num_steps)

    def apply_transformation(self, structure: Structure) -> dict[str, Any]:
        """Applies rigid displacements to create GSFE/USFE structures.

        Args:
            structure (Structure): Input structure used as the reference state.

        Returns:
            dict[str, dict[float, Structure] | float]: A dictionary with keys ``"displaced_structures"``, mapping displacement
                fractions to the corresponding displaced `Structure`, and ``"fault_area"``, the fault-plane area in Angstrom squared.
        """
        displaced_structures: dict[float, Structure] = {}

        normal_cart = self._plane_normal_cart(structure)
        slip_dir_cart = self._slip_direction_cart(structure, normal_cart)
        burgers_vector = 0.5 * slip_dir_cart
        fault_area = self._fault_area(structure, normal_cart)

        cart_coords = structure.cart_coords
        plane_projection = cart_coords @ normal_cart
        plane_cut = float(np.median(plane_projection))
        upper_mask = plane_projection > plane_cut

        for frac in self.displacement_fractions:
            displaced = structure.copy()
            displaced.translate_sites(
                indices=np.where(upper_mask)[0].tolist(),
                vector=frac * burgers_vector,
                frac_coords=False,
                to_unit_cell=False,
            )
            displaced_structures[float(frac)] = displaced

        return {"displaced_structures": displaced_structures, "fault_area": fault_area}

    def _plane_normal_cart(self, structure: Structure) -> np.ndarray:
        """Return a normalized Cartesian plane normal.

        Args:
            structure: Reference structure.

        Returns:
            Cartesian unit vector normal to the selected slip plane.
        """
        miller = self._SLIP_SYSTEMS[self.slip_plane]["plane"]
        normal = structure.lattice.reciprocal_lattice_crystallographic.get_cartesian_coords(miller)
        return normal / np.linalg.norm(normal)

    def _slip_direction_cart(self, structure: Structure, normal_cart: np.ndarray) -> np.ndarray:
        """Return a normalized in-plane Cartesian slip direction.

        Args:
            structure: Reference structure.
            normal_cart: Cartesian unit normal for the selected slip plane.

        Returns:
            Cartesian unit vector of the in-plane slip direction.
        """
        direction_hkl = self._SLIP_SYSTEMS[self.slip_plane]["direction"]
        direction = structure.lattice.get_cartesian_coords(direction_hkl)
        direction = direction - np.dot(direction, normal_cart) * normal_cart
        return direction / np.linalg.norm(direction)

    @staticmethod
    def _fault_area(structure: Structure, normal_cart: np.ndarray) -> float:
        """Calculate the cross-sectional area of the fault plane.

        Args:
            structure: Reference structure.
            normal_cart: Cartesian unit normal for the selected slip plane.

        Returns:
            Fault-plane area in Angstrom squared.
        """
        lattice_matrix = structure.lattice.matrix
        projections = np.abs(lattice_matrix @ normal_cart)
        normal_axis = int(np.argmax(projections))
        in_plane_axes = [i for i in range(3) if i != normal_axis]
        v1 = lattice_matrix[in_plane_axes[0]]
        v2 = lattice_matrix[in_plane_axes[1]]
        return float(np.linalg.norm(np.cross(v1, v2)))
