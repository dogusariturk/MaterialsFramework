"""Generates structures for surface binding energy (SBE) calculations.

Produces the structures the SBE workflow needs from a relaxed bulk structure: inequivalent slab
terminations across Miller indices up to a maximum index, a larger supercell built from a chosen
(relaxed) slab termination, single-atom vacancy structures at that supercell's surface sites, and an
isolated-atom reference structure for a given element, for use in `SBEAnalyzer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import generate_all_slabs

if TYPE_CHECKING:
    from pymatgen.core.surface import Slab

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class SBETransformation:
    """Generates slab, supercell, vacancy, and isolated-atom structures for SBE calculations.

    `apply_transformation` is the sole public method. Exactly one of its `structure`, `slab`,
    `supercell_slab`, or `element` arguments should be given; which one selects the structure
    generated, since each is built from the previous step's (calculator-relaxed) output rather
    than from the original bulk structure.
    """

    def __init__(
        self,
        max_index: int = 1,
        min_slab_size: float = 10.0,
        min_vacuum_size: float = 10.0,
        height: float = 1.0,
        supercell_size: list[int] | None = None,
        isolated_atom_box_size: float = 20.0,
    ) -> None:
        """Initializes the `SBETransformation` object.

        Args:
            max_index (int, optional): Maximum Miller index to consider when generating slabs. Defaults to 1.
            min_slab_size (float, optional): Minimum slab thickness in Angstroms for slab generation. Defaults to 10.0.
            min_vacuum_size (float, optional): Minimum vacuum size in Angstroms for slab generation. Defaults to 10.0.
            height (float, optional): Height above the surface, in Angstroms, used to identify surface atoms via
                `pymatgen.analysis.adsorption.AdsorbateSiteFinder`. Defaults to 1.0.
            supercell_size (list[int] | None, optional): Supercell replication factors applied to a slab
                termination before vacancy generation. Defaults to ``[4, 4, 1]``.
            isolated_atom_box_size (float, optional): Side length, in Angstroms, of the cubic cell used to
                approximate a non-interacting isolated atom. Defaults to 20.0.
        """
        self.max_index = max_index
        self.min_slab_size = min_slab_size
        self.min_vacuum_size = min_vacuum_size
        self.height = height
        self.supercell_size = supercell_size if supercell_size is not None else [4, 4, 1]
        self.isolated_atom_box_size = isolated_atom_box_size

    def apply_transformation(
        self,
        structure: Structure | None = None,
        slab: Structure | None = None,
        supercell_slab: Structure | None = None,
        element: str | None = None,
    ) -> list[Slab] | Structure | list[dict[str, int | str | Structure]]:
        """Generates one of the SBE pipeline structures from exactly one of the given inputs.

        Args:
            structure (Structure | None, optional): The (relaxed) bulk structure to generate slab
                terminations from.
            slab (Structure | None, optional): The slab termination, typically already relaxed, to
                replicate into a supercell.
            supercell_slab (Structure | None, optional): The slab supercell to generate one
                single-atom vacancy structure from per identified surface site.
            element (str | None, optional): The chemical symbol to build an isolated single-atom
                reference structure for.

        Returns:
            list[Slab] | Structure | list[dict[str, int | str | Structure]]: Depending on which
                argument was given:
                - `structure`: The generated slab terminations, repaired to remove broken bonds at
                    the surface. Each slab's `miller_index` attribute identifies which Miller index
                    it belongs to.
                - `slab`: The replicated supercell, independent of `slab`.
                - `supercell_slab`: One entry per surface site, each a dictionary with keys
                    `site_index` (the removed site's index in `supercell_slab`), `element` (its
                    species string), and `structure` (a copy of `supercell_slab` with that site
                    removed).
                - `element`: A single atom of `element` in a cubic cell of side
                    `self.isolated_atom_box_size` Angstroms, large enough to make periodic-image
                    interactions negligible.

        Raises:
            ValueError: If zero, or more than one, of `structure`, `slab`, `supercell_slab`, and
                `element` is given.
        """
        given = [
            name
            for name, value in (
                ("structure", structure),
                ("slab", slab),
                ("supercell_slab", supercell_slab),
                ("element", element),
            )
            if value is not None
        ]
        if len(given) != 1:
            raise ValueError("Exactly one of `structure`, `slab`, `supercell_slab`, or `element` must be given.")

        if structure is not None:
            return generate_all_slabs(
                structure=structure,
                max_index=self.max_index,
                min_slab_size=self.min_slab_size,
                min_vacuum_size=self.min_vacuum_size,
                repair=True,
            )
        if slab is not None:
            return slab.make_supercell(self.supercell_size, in_place=False)
        if supercell_slab is not None:
            return self._generate_vacancy_structures(supercell_slab)
        if element is not None:
            return self._isolated_atom_structure(element)
        raise ValueError("Exactly one of `structure`, `slab`, `supercell_slab`, or `element` must be given.")

    def _generate_vacancy_structures(self, supercell_slab: Structure) -> list[dict[str, int | str | Structure]]:
        """Creates one single-atom vacancy structure per identified surface site.

        Surface sites are identified with `pymatgen.analysis.adsorption.AdsorbateSiteFinder`, which flags
        sites within `self.height` Angstroms of the topmost site along the surface normal.

        Args:
            supercell_slab (Structure): The slab supercell to remove surface atoms from.

        Returns:
            list[dict[str, int | str | Structure]]: One entry per surface site, each a dictionary with keys
                ``site_index`` (the removed site's index in `supercell_slab`), ``element`` (its species string),
                and ``structure`` (a copy of `supercell_slab` with that site removed).
        """
        site_finder = AdsorbateSiteFinder(slab=supercell_slab, height=self.height)

        vacancy_structures = []
        for site_index, site in enumerate(site_finder.slab):
            if site.properties.get("surface_properties") != "surface":
                continue
            vacancy_structure = supercell_slab.copy()
            vacancy_structure.remove_sites([site_index])
            vacancy_structures.append(
                {
                    "site_index": site_index,
                    "element": site.species_string,
                    "structure": vacancy_structure,
                }
            )
        return vacancy_structures

    def _isolated_atom_structure(self, element: str) -> Structure:
        """Builds a single-atom structure approximating a non-interacting isolated atom.

        Args:
            element (str): The chemical symbol of the isolated atom.

        Returns:
            Structure: A single atom of `element` in a cubic cell of side `self.isolated_atom_box_size`
                Angstroms, large enough to make periodic-image interactions negligible.
        """
        lattice = Lattice.cubic(self.isolated_atom_box_size)
        return Structure(lattice, [element], [[0.0, 0.0, 0.0]])
