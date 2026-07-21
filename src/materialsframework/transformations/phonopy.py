"""Generates distorted structures for Phonopy calculations.

Produces supercells with atomic displacements needed to compute force constants and
phonon properties: vibrational modes, thermal properties, and lattice dynamics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from phonopy import Phonopy
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class PhonopyDisplacementTransformation:
    """Generates displaced structures for Phonopy calculations.

    Creates supercells with atomic displacements for computing phonon spectra, thermal
    conductivity, and other lattice-dynamical properties.
    """

    def apply_transformation(
        self,
        structure: Structure,
        distance: float = 0.01,
        supercell_matrix: list | None = None,
        primitive_matrix: list | None = None,
        log_level: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate displaced supercells for Phonopy calculations.

        Args:
            structure (Structure): The input structure to be displaced.
            distance (float, optional): The maximum atomic displacement distance. Defaults to 0.01.
            supercell_matrix (list, optional): The supercell matrix to generate supercells for phonon calculations.
                Defaults to a 2x2x2 supercell.
            primitive_matrix (list, optional): The primitive matrix to generate the primitive cell. Defaults to None.
            log_level (int, optional): The log level for Phonopy. Defaults to 0.
            **kwargs: Additional keyword arguments for the `Phonopy.generate_displacement` method.

        Returns:
            dict[str, Phonopy | list[Structure] | np.ndarray | list]: Dictionary with keys:
                - ``phonon``: The `Phonopy` object used to generate the displaced structures.
                - ``displaced_structures``: The list of displaced structures for phonon calculations.
                - ``displacements``: The displacement vectors used to generate the displaced structures.
        """
        try:
            from phonopy import Phonopy
            from pymatgen.io.phonopy import get_phonopy_structure
        except ImportError as e:
            raise ImportError("phonopy is required. Install it with: pip install materialsframework[phonopy]") from e

        supercell_matrix = np.diag(supercell_matrix) if supercell_matrix else np.diag([2, 2, 2])

        phonopy_structure = get_phonopy_structure(structure)

        phonon = Phonopy(
            unitcell=phonopy_structure,
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            log_level=log_level,
        )

        displaced_structures = self._get_displaced_structures(phonon, distance=distance, **kwargs)
        displacements = phonon.displacements

        return {"phonon": phonon, "displaced_structures": displaced_structures, "displacements": displacements}

    def _get_displaced_structures(self, phonon: Phonopy, distance: float = 0.01, **kwargs) -> list[Structure]:
        """Generate displaced structures using Phonopy.

        Args:
            phonon (Phonopy): The `Phonopy` object to generate displaced structures for.
            distance (float, optional): The maximum atomic displacement distance. Defaults to 0.01.
            **kwargs: Additional keyword arguments for `Phonopy.generate_displacements`.

        Returns:
            list[Structure]: A list of displaced structures for phonon calculations.
        """
        try:
            from pymatgen.io.phonopy import get_pmg_structure
        except ImportError as e:
            raise ImportError("phonopy is required. Install it with: pip install materialsframework[phonopy]") from e

        phonon.generate_displacements(distance=distance, **kwargs)

        displaced_supercells = phonon.supercells_with_displacements
        displaced_structures = [get_pmg_structure(cell) for cell in displaced_supercells if cell is not None]

        return displaced_structures
