"""Generates distorted structures for Phono3py calculations.

Produces supercells with atomic displacements needed to compute second- and third-order
force constants for anharmonic phonon and thermal-conductivity calculations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from phono3py import Phono3py
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class Phono3pyDisplacementTransformation:
    """Generates displaced structures for Phono3py calculations.

    Creates supercells with atomic displacements for both second- and third-order force
    constants.
    """

    def __init__(self) -> None:
        """Initializes the `Phono3pyDisplacementTransformation` object."""

    def apply_transformation(
        self,
        structure: Structure,
        distance: float = 0.03,
        supercell_matrix: list | None = None,
        primitive_matrix: list | str = "auto",
        phonon_supercell_matrix: list | None = None,
        log_level: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate displaced supercells for Phono3py calculations.

        Args:
            structure (Structure): The input structure to be used for generating displacements.
            distance (float, optional): The maximum atomic displacement distance. Defaults to 0.03.
            supercell_matrix (list, optional): The supercell matrix for third-order force constant calculations.
                Defaults to a 2x2x2 supercell.
            primitive_matrix (list | str, optional): The primitive matrix for the supercell. Defaults to 'auto'.
            phonon_supercell_matrix (list, optional): The supercell matrix for second-order force constant
                calculations. Defaults to a 3x3x3 supercell.
            log_level (int, optional): The log level for Phono3py. Defaults to 0.
            **kwargs: Additional keyword arguments for the `Phono3py.generate_displacement` method.

        Returns:
            dict[str, Phono3py | list[Structure]]: Dictionary with keys:
                - ``phonon``: The `Phono3py` object used to generate the displacements.
                - ``phonon_supercells_with_displacements``: Displaced supercells for phonon (second-order) force
                    constant calculations.
                - ``supercells_with_displacements``: Displaced supercells for third-order force constant calculations.
                - ``phonon_displacements``: The atomic displacements for the phonon supercells.
                - ``supercell_displacements``: The atomic displacements for the third-order force-constant supercells.
        """
        try:
            from phono3py import Phono3py
            from pymatgen.io.phonopy import get_phonopy_structure
        except ImportError as e:
            raise ImportError("phono3py is required. Install it with: pip install materialsframework[phono3py]") from e

        supercell_matrix = np.diag(supercell_matrix) if supercell_matrix else np.diag([2, 2, 2])
        phonon_supercell_matrix = np.diag(phonon_supercell_matrix) if phonon_supercell_matrix else np.diag([3, 3, 3])

        phonopy_structure = get_phonopy_structure(structure)

        phonon = Phono3py(
            unitcell=phonopy_structure,
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            phonon_supercell_matrix=phonon_supercell_matrix,
            log_level=log_level,
        )

        (
            phonon_supercells_with_displacements,
            supercells_with_displacements,
        ) = self._get_displaced_structures(phonon, distance=distance, **kwargs)

        phonon_displacements = phonon.phonon_displacements
        supercell_displacements = phonon.displacements

        return {
            "phonon": phonon,
            "phonon_supercells_with_displacements": phonon_supercells_with_displacements,
            "supercells_with_displacements": supercells_with_displacements,
            "phonon_displacements": phonon_displacements,
            "supercell_displacements": supercell_displacements,
        }

    def _get_displaced_structures(
        self,
        phonon: Phono3py,
        distance: float = 0.03,
        is_plusminus: bool | str = "auto",
        is_diagonal: bool = True,
    ) -> tuple[list[Structure], list[Structure]]:
        """Generate displaced structures using Phono3py.

        Args:
            phonon (Phono3py): The `Phono3py` object to generate the displacements for.
            distance (float, optional): The maximum atomic displacement distance. Defaults to 0.03.
            is_plusminus (bool | str, optional): Whether to generate both positive and negative displacements.
                Defaults to "auto".
            is_diagonal (bool, optional): Whether to only displace atoms along diagonal directions. Defaults to True.

        Returns:
            tuple[list[Structure], list[Structure]]: Two lists of displaced structures for phonon (second-order) and third-order
                force constant calculations.
        """
        try:
            from pymatgen.io.phonopy import get_pmg_structure
        except ImportError as e:
            raise ImportError("phono3py is required. Install it with: pip install materialsframework[phono3py]") from e

        phonon.generate_displacements(distance=distance, is_plusminus=is_plusminus, is_diagonal=is_diagonal)

        displaced_supercells = phonon.supercells_with_displacements
        displaced_structures = [get_pmg_structure(cell) for cell in displaced_supercells if cell is not None]

        phonon.generate_fc2_displacements(distance=distance, is_plusminus=is_plusminus, is_diagonal=is_diagonal)

        displaced_phonon_supercells = phonon.phonon_supercells_with_displacements
        displaced_phonon_structures = [get_pmg_structure(cell) for cell in displaced_phonon_supercells if cell is not None]

        return displaced_phonon_structures, displaced_structures
