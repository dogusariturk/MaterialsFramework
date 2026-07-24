"""Generates structures for formation energy calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.data import atomic_numbers, reference_states
from pymatgen.core import Element

from materialsframework.utils import to_structure

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class FormationEnergyTransformation:
    """Generates elemental reference structures for formation energy calculations.

    For each element, the experimentally-tabulated ground-state prototype (``ase.data.reference_states``,
    the same table ``ase.build.bulk`` consults) is used as the elemental reference: molecular gas-phase
    elements (H, N, O, F) become an isolated dimer, monatomic gas-phase elements (noble gases) become an
    isolated atom, and everything else with a simple Bravais-lattice ground state (FCC, BCC, HCP, diamond
    cubic, rhombohedral, body-centered tetragonal, simple cubic, ...) is built directly from its known
    symmetry and lattice parameter. In all of these cases a single candidate is returned, since the
    reference structure is already known rather than guessed.

    A handful of elements have a ground state that requires an explicit atomic basis ``ase.build.bulk``
    cannot construct from a formula alone (e.g. Mn, P, S, Ga), or have no tabulated reference state at all
    (mostly radioactive actinides). For these, several candidate high-symmetry lattices (FCC, BCC, HCP,
    diamond, simple cubic) are estimated from the element's atomic radius; the analyzer relaxes all
    candidates with the same MLIP and selects the lowest-energy phase as the elemental reference.
    """

    def apply_transformation(self, structure: Atoms | Structure) -> list[tuple[list[Structure], int]]:
        """Apply the transformation to generate elemental reference structures.

        For each element present in ``structure``, either its known experimental ground-state
        structure is built directly, or (if that ground state cannot be constructed from a
        formula alone) several candidate high-symmetry lattices are generated from the element's
        empirical atomic radius. See the class docstring for details.

        Args:
            structure (Atoms | Structure): The compound structure whose composition
                determines which elemental references are generated.

        Returns:
            list[tuple[list[Structure], int]]: A list of ``(candidates, n_atoms)`` tuples, where ``candidates``
                is a list of one or more candidate pymatgen ``Structure`` references and ``n_atoms`` is the
                count of that element in the compound.
        """
        pure_structures: list[tuple[list[Structure], int]] = []

        structure = to_structure(structure)

        for element, num in structure.composition.get_el_amt_dict().items():
            candidates = self._reference_candidates(element)
            pure_structures.append((candidates, int(num)))

        return pure_structures

    def _reference_candidates(self, element: str) -> list[Structure]:
        """Return the candidate reference structure(s) for a single element.

        Returns a single-item list when the element's experimental ground state is known and
        unambiguous, or several candidate lattices when it must be guessed.

        Args:
            element (str): The element symbol to generate reference structure(s) for.

        Returns:
            list[Structure]: One or more candidate pymatgen ``Structure`` references.
        """
        z = atomic_numbers.get(element)
        ref = reference_states[z] if z is not None else None
        symmetry = ref["symmetry"] if ref else None

        if symmetry == "diatom":
            return [to_structure(self._boxed(molecule(f"{element}2")))]
        if symmetry == "atom":
            return [to_structure(self._boxed(Atoms(element)))]
        if symmetry is not None:
            try:
                return [to_structure(bulk(element))]
            except (ValueError, RuntimeError):
                pass

        return self._guessed_candidates(element)

    @staticmethod
    def _boxed(atoms: Atoms) -> Atoms:
        """Place an isolated molecule/atom in a large periodic vacuum box.

        Args:
            atoms (Atoms): The isolated molecule or atom to box.

        Returns:
            Atoms: The same ``Atoms`` object, centered in a cubic vacuum cell.
        """
        vacuum_box = 20.0

        atoms.set_cell([vacuum_box] * 3)
        atoms.center()
        atoms.pbc = True
        return atoms

    @staticmethod
    def _guessed_candidates(element: str) -> list[Structure]:
        """Generate candidate FCC/BCC/HCP/diamond/simple-cubic lattices from the atomic radius.

        Args:
            element (str): The element symbol to generate candidate structures for.

        Returns:
            list[Structure]: Candidate pymatgen ``Structure`` references to relax and compare.
        """
        fallback_radius = 1.4

        r = float(Element(element).atomic_radius or fallback_radius)

        a_fcc = 2 * r * np.sqrt(2)
        a_bcc = 4 * r / np.sqrt(3)
        a_hcp = 2 * r
        c_hcp = a_hcp * np.sqrt(8 / 3)
        a_diamond = 8 * r / np.sqrt(3)
        a_sc = 2 * r

        return [
            to_structure(bulk(element, "fcc", a=a_fcc)),
            to_structure(bulk(element, "bcc", a=a_bcc)),
            to_structure(bulk(element, "hcp", a=a_hcp, c=c_hcp)),
            to_structure(bulk(element, "diamond", a=a_diamond)),
            to_structure(bulk(element, "sc", a=a_sc)),
        ]
