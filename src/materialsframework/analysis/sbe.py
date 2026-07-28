"""This module provides a class to compute surface binding energies (SBE) for a bulk structure.

The `SBEAnalyzer` class relaxes the bulk structure, screens slab terminations (via `SBETransformation`)
across Miller indices up to a maximum index to find the lowest-surface-energy termination, then creates
a single-atom vacancy at each surface site of a supercell built from that termination. The surface
binding energy for a site is ``E_a + E_{s+v} - E_s``, where ``E_a`` is the isolated-atom energy, ``E_s``
is the perfect supercell slab energy, and ``E_{s+v}`` is the energy of the supercell slab with that site
removed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from materialsframework.analysis.base import BaseAnalyzer
from materialsframework.analysis.utils import require_properties
from materialsframework.transformations.sbe import SBETransformation
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure
    from pymatgen.core.surface import Slab

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class SBEAnalyzer(BaseAnalyzer):
    """A class used to compute surface binding energies (SBE) for a bulk structure."""

    def __init__(
        self,
        max_index: int = 1,
        min_slab_size: float = 10.0,
        min_vacuum_size: float = 10.0,
        height: float = 1.0,
        supercell_size: list[int] | None = None,
        calculator: BaseCalculator | None = None,
        sbe_transformation: SBETransformation | None = None,
    ) -> None:
        """Initializes the `SBEAnalyzer` object.

        Args:
            max_index (int, optional): Maximum Miller index to consider when generating slabs. Defaults to 1.
            min_slab_size (float, optional): Minimum slab thickness in Angstroms for slab generation. Defaults to 10.0.
            min_vacuum_size (float, optional): Minimum vacuum size in Angstroms for slab generation. Defaults to 10.0.
            height (float, optional): Height above the surface, in Angstroms, used to identify surface atoms.
                Defaults to 1.0.
            supercell_size (list[int] | None, optional): Supercell replication factors used when building the
                slab supercell for vacancy calculations. Defaults to ``[4, 4, 1]``.
            calculator (BaseCalculator | None, optional): The calculator used for energy calculations. Defaults
                to a lazily constructed default calculator.
            sbe_transformation (SBETransformation | None, optional): The transformation object used to generate
                slabs, supercells, vacancy structures, and isolated-atom references. If not provided, a new
                instance is initialized from `max_index`, `min_slab_size`, `min_vacuum_size`, `height`, and
                `supercell_size`.
        """
        super().__init__(calculator)
        self.max_index = max_index
        self.min_slab_size = min_slab_size
        self.min_vacuum_size = min_vacuum_size
        self.height = height
        self.supercell_size = supercell_size if supercell_size is not None else [4, 4, 1]

        self._sbe_transformation = sbe_transformation
        self._isolated_atom_energy_cache: dict[str, float] = {}

    @require_properties("energy")
    def calculate(self, structure: Structure | Atoms, is_relaxed: bool = False) -> dict[str, Any]:
        """Calculates the surface binding energy (SBE) for a given bulk structure.

        Every energy/metadata collection below is a flat list of uniform dicts, joinable on
        ``miller_index``/``termination_index``/``site_index``. `Structure` objects are kept out of
        these records and returned separately under ``structures``, keyed the same way, so the main
        result stays small and easy to hand to e.g. `pandas.DataFrame`.

        Args:
            structure (Structure | Atoms): The bulk structure to be analyzed.
            is_relaxed (bool, optional): Whether the input structure is already relaxed. Defaults to False.

        Returns:
            dict[str, Any]: A dictionary with the following keys:
                - ``bulk_energy_per_atom``: The energy per atom of the (relaxed) bulk structure.
                - ``best_miller_index``: The Miller index containing the single lowest-surface-energy termination.
                - ``best_surface_energy``: That termination's ``surface_energy`` (the minimum across every
                    entry in ``surface_energies``).
                - ``surface_energies``: One entry per screened slab termination (every Miller index up to
                    `max_index`), each a dict with keys ``miller_index``, ``termination_index``, ``slab_area``,
                    ``slab_energy``, and ``surface_energy``.
                - ``isolated_atom_energies``: A dict mapping each element symbol to its isolated-atom energy.
                - ``terminations``: One entry per termination of ``best_miller_index``, each a dict with keys
                    ``miller_index``, ``termination_index``, ``supercell_slab_energy`` (E_s), and
                    ``avg_surface_binding_energy_by_element`` (mean SBE per element for that termination alone).
                - ``vacancy_results``: One entry per surface site of every termination of ``best_miller_index``,
                    each a dict with keys ``miller_index``, ``termination_index``, ``site_index``, ``element``,
                    ``vacancy_energy`` (E_{s+v}), and ``surface_binding_energy``.
                - ``avg_surface_binding_energy_by_element``: Mean, across terminations, of each termination's
                    per-element average SBE (each termination weighted equally, not each site).
                - ``avg_surface_binding_energy``: Mean, across terminations and elements, of the per-termination
                    per-element averages above.
                - ``structures``: A dict with keys ``bulk_structure`` (the relaxed bulk `Structure`), ``slabs``
                    (parallel to ``surface_energies``, each with ``slab`` and ``relaxed_slab``), ``supercells``
                    (parallel to ``terminations``, each with ``supercell_slab``), and ``vacancies`` (parallel to
                    ``vacancy_results``, each with ``structure`` (unrelaxed) and ``relaxed_structure``).

        Raises:
            ValueError: If the calculator does not implement the 'energy' property, or if no slabs were
                generated for the given structure and parameters.
        """
        structure = self._ensure_relaxed(structure, is_relaxed)

        bulk_energy_per_atom = self.calculator.calculate(structure)["energy"] / structure.num_sites

        slabs = self.sbe_transformation.apply_transformation(structure)
        surface_energies, slab_structures, best_miller_index, best_surface_energy = self._screen_miller_indices(
            slabs, bulk_energy_per_atom
        )
        if best_miller_index is None:
            raise ValueError("No slabs were generated for the given structure and parameters.")

        isolated_atom_energies = {
            element.symbol: self._isolated_atom_energy(element.symbol) for element in structure.composition.elements
        }

        # termination_index is assigned in encounter order, so filtering preserves it without an explicit sort.
        best_slabs = [entry for entry in slab_structures if entry["miller_index"] == best_miller_index]
        termination_payloads = [
            self._evaluate_termination(
                entry["miller_index"], entry["termination_index"], entry["relaxed_slab"], isolated_atom_energies
            )
            for entry in best_slabs
        ]

        terminations = [payload["termination"] for payload in termination_payloads]
        supercell_structures = [payload["supercell_structure"] for payload in termination_payloads]
        vacancy_results = [result for payload in termination_payloads for result in payload["vacancy_results"]]
        vacancy_structures = [entry for payload in termination_payloads for entry in payload["vacancy_structures"]]

        per_element_means: dict[str, list[float]] = defaultdict(list)
        for termination in terminations:
            for element, mean_value in termination["avg_surface_binding_energy_by_element"].items():
                per_element_means[element].append(mean_value)
        avg_by_element = {element: float(np.mean(values)) for element, values in per_element_means.items()}
        all_means = [value for values in per_element_means.values() for value in values]
        avg_overall = float(np.mean(all_means)) if all_means else None

        return {
            "bulk_energy_per_atom": bulk_energy_per_atom,
            "best_miller_index": best_miller_index,
            "best_surface_energy": best_surface_energy,
            "surface_energies": surface_energies,
            "isolated_atom_energies": isolated_atom_energies,
            "terminations": terminations,
            "vacancy_results": vacancy_results,
            "avg_surface_binding_energy_by_element": avg_by_element,
            "avg_surface_binding_energy": avg_overall,
            "structures": {
                "bulk_structure": structure,
                "slabs": slab_structures,
                "supercells": supercell_structures,
                "vacancies": vacancy_structures,
            },
        }

    def _screen_miller_indices(
        self,
        slabs: list[Slab],
        bulk_energy_per_atom: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, ...] | None, float | None]:
        """Relaxes every generated slab (cell fixed) into flat energy and structure records.

        Args:
            slabs (list[Slab]): The slabs generated by `sbe_transformation.apply_transformation`.
            bulk_energy_per_atom (float): The bulk structure's energy per atom, used as the surface-energy reference.

        Returns:
            tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, ...] | None, float | None]: The
                ``surface_energies`` records, the parallel ``structures["slabs"]`` records (see `calculate`), the
                Miller index containing the single lowest ``surface_energy`` termination, and that termination's
                ``surface_energy`` (both ``None`` if `slabs` is empty).
        """
        prev_relax_cell = self.calculator.relax_cell
        self.calculator.relax_cell = False
        try:
            surface_energies: list[dict[str, Any]] = []
            slab_structures: list[dict[str, Any]] = []
            termination_counters: dict[tuple[int, ...], int] = {}
            best_miller_index = None
            best_gamma = None
            for slab in slabs:
                slab_area = slab.surface_area
                relaxed = self.calculator.relax(slab)
                relaxed_slab = relaxed["final_structure"]
                slab_energy = relaxed["energy"]
                gamma = (slab_energy - relaxed_slab.num_sites * bulk_energy_per_atom) / (2 * slab_area)

                miller_index = tuple(slab.miller_index)
                termination_index = termination_counters.get(miller_index, 0)
                termination_counters[miller_index] = termination_index + 1

                surface_energies.append(
                    {
                        "miller_index": miller_index,
                        "termination_index": termination_index,
                        "slab_area": slab_area,
                        "slab_energy": slab_energy,
                        "surface_energy": gamma,
                    }
                )
                slab_structures.append(
                    {
                        "miller_index": miller_index,
                        "termination_index": termination_index,
                        "slab": slab,
                        "relaxed_slab": relaxed_slab,
                    }
                )

                if best_gamma is None or gamma < best_gamma:
                    best_gamma = gamma
                    best_miller_index = miller_index
        finally:
            self.calculator.relax_cell = prev_relax_cell

        return surface_energies, slab_structures, best_miller_index, best_gamma

    def _evaluate_termination(
        self,
        miller_index: tuple[int, ...],
        termination_index: int,
        relaxed_slab: Structure,
        isolated_atom_energies: dict[str, float],
    ) -> dict[str, Any]:
        """Builds a supercell for one termination and computes the SBE at each of its surface sites.

        Args:
            miller_index (tuple[int, ...]): The Miller index `relaxed_slab` belongs to.
            termination_index (int): This termination's index within `miller_index`.
            relaxed_slab (Structure): The relaxed (cell-fixed) slab termination to evaluate.
            isolated_atom_energies (dict[str, float]): Isolated-atom energy per element symbol.

        Returns:
            dict[str, Any]: A dict with keys ``termination`` (a `calculate` ``terminations`` entry),
                ``supercell_structure`` (the parallel ``structures["supercells"]`` entry), ``vacancy_results``
                (a list of `calculate` ``vacancy_results`` entries for this termination), and ``vacancy_structures``
                (the parallel ``structures["vacancies"]`` entries).
        """
        supercell_slab = self.sbe_transformation.apply_transformation(slab=relaxed_slab)
        supercell_slab_energy = self.calculator.calculate(supercell_slab)["energy"]

        vacancy_results: list[dict[str, Any]] = []
        vacancy_structures: list[dict[str, Any]] = []
        prev_relax_cell = self.calculator.relax_cell
        self.calculator.relax_cell = False
        try:
            for vacancy in self.sbe_transformation.apply_transformation(supercell_slab=supercell_slab):
                relaxed = self.calculator.relax(vacancy["structure"])
                vacancy_energy = relaxed["energy"]
                surface_binding_energy = isolated_atom_energies[vacancy["element"]] + vacancy_energy - supercell_slab_energy
                vacancy_results.append(
                    {
                        "miller_index": miller_index,
                        "termination_index": termination_index,
                        "site_index": vacancy["site_index"],
                        "element": vacancy["element"],
                        "vacancy_energy": vacancy_energy,
                        "surface_binding_energy": surface_binding_energy,
                    }
                )
                vacancy_structures.append(
                    {
                        "miller_index": miller_index,
                        "termination_index": termination_index,
                        "site_index": vacancy["site_index"],
                        "structure": vacancy["structure"],
                        "relaxed_structure": relaxed["final_structure"],
                    }
                )
        finally:
            self.calculator.relax_cell = prev_relax_cell

        per_element_energies: dict[str, list[float]] = defaultdict(list)
        for vacancy_result in vacancy_results:
            per_element_energies[vacancy_result["element"]].append(vacancy_result["surface_binding_energy"])
        avg_by_element = {element: float(np.mean(values)) for element, values in per_element_energies.items()}

        return {
            "termination": {
                "miller_index": miller_index,
                "termination_index": termination_index,
                "supercell_slab_energy": supercell_slab_energy,
                "avg_surface_binding_energy_by_element": avg_by_element,
            },
            "supercell_structure": {
                "miller_index": miller_index,
                "termination_index": termination_index,
                "supercell_slab": supercell_slab,
            },
            "vacancy_results": vacancy_results,
            "vacancy_structures": vacancy_structures,
        }

    def _isolated_atom_energy(self, element: str) -> float:
        """Returns the isolated-atom energy for `element`, computing and caching it on first use.

        Args:
            element (str): The chemical symbol to compute the isolated-atom energy for.

        Returns:
            float: The isolated-atom energy, cached on this analyzer instance so repeated `calculate()`
                calls (even for different structures) don't re-run the calculation.
        """
        if element not in self._isolated_atom_energy_cache:
            isolated_atom = self.sbe_transformation.apply_transformation(element=element)
            self._isolated_atom_energy_cache[element] = self.calculator.calculate(isolated_atom)["energy"]
        return self._isolated_atom_energy_cache[element]

    @lazy_property("_sbe_transformation")
    def sbe_transformation(self) -> SBETransformation:
        """Returns the transformation object used to generate slabs, supercells, and vacancy structures.

        Returns:
            SBETransformation: The transformation object used to generate structures.
        """
        return SBETransformation(
            max_index=self.max_index,
            min_slab_size=self.min_slab_size,
            min_vacuum_size=self.min_vacuum_size,
            height=self.height,
            supercell_size=self.supercell_size,
        )
