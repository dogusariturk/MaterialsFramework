"""Generates structures for formation energy calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.build import bulk
from pymatgen.core import Element

from materialsframework.utils import to_structure

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

_FALLBACK_RADIUS = 1.4  # Å; fallback for elements without a tabulated atomic radius


class FormationEnergyTransformation:
    """Generates elemental reference structures for formation energy calculations.

    For each element in the compound, this produces three candidate crystal structures
    (FCC, BCC, HCP) using the element's atomic radius to estimate lattice parameters. The
    analyzer relaxes all candidates with the same MLIP and selects the lowest-energy phase
    as the elemental reference.
    """

    def apply_transformation(self, structure: Atoms | Structure) -> list[tuple[list[Structure], int]]:
        """Apply the transformation to generate elemental reference structures.

        For each element present in ``structure``, three candidate crystal structures
        (FCC, BCC, HCP) are generated using the element's empirical atomic radius to
        estimate lattice parameters.

        Args:
            structure (Atoms | Structure): The compound structure whose composition
                determines which elemental references are generated.

        Returns:
            list[tuple[list[Structure], int]]: A list of ``(candidates, n_atoms)`` tuples, where ``candidates`` is a list of
                FCC/BCC/HCP pymatgen ``Structure`` objects and ``n_atoms`` is the count of that element in the compound.
        """
        pure_structures: list[tuple[list[Structure], int]] = []

        structure = to_structure(structure)

        for element, num in structure.composition.get_el_amt_dict().items():
            r = float(Element(element).atomic_radius or _FALLBACK_RADIUS)

            a_fcc = 2 * r * np.sqrt(2)
            a_bcc = 4 * r / np.sqrt(3)
            a_hcp = 2 * r
            c_hcp = a_hcp * np.sqrt(8 / 3)

            candidates = [
                to_structure(bulk(element, "fcc", a=a_fcc)),
                to_structure(bulk(element, "bcc", a=a_bcc)),
                to_structure(bulk(element, "hcp", a=a_hcp, c=c_hcp)),
            ]

            pure_structures.append((candidates, int(num)))

        return pure_structures
