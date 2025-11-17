"""This module implements the SBEAnalyzer class for computing surface binding energies (SBE) for a given bulk structure.

The SBEAnalyzer class calculates the surface binding energy by following a systematic workflow that includes:
1. Relaxing the bulk structure (if not already relaxed).
2. Generating inequivalent slabs with Miller indices up to a specified maximum.
3. Relaxing each slab termination and calculating the surface energy (gamma).
4. Selecting the Miller index with the lowest surface energy and building supercells for its terminations
5. Identifying surface sites, creating single-atom vacancies at those sites, and calculating vacancy energies.
6. Computing isolated-atom energies and aggregating SBE per element and termination.

Notes:
    - The SBE definition used throughout is::

        SBE = E_a + E_{s+v} - E_s

      where E_a is the isolated atom energy, E_s is the perfect supercell slab energy and
    E_{s+v} is the energy of the slab with a single surface vacancy.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import Slab, generate_all_slabs
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class SBEAnalyzer:
    """Analyzer for computing surface binding energies (SBE).

    The SBEAnalyzer encapsulates the following high-level steps:

    1. (Optional) relax the bulk structure using the provided calculator.
    2. Generate inequivalent slabs up to ``max_index``.
    3. Relax each slab termination and compute the surface energy (gamma).
    4. Select the Miller index with the lowest surface energy and build supercells for its terminations.
    5. Identify surface sites, create single-atom vacancies at those sites and compute vacancy energies.
    6. Compute isolated-atom energies and aggregate SBE per element and termination.

    where:
    - E_a: total energy of the isolated sputtered atom
    - E_s: total energy of the perfect crystal slab
    - E_{s+v}: energy of the slab with one surface vacancy
    """

    def __init__(
            self,
            max_index: int = 1,
            min_slab_size: float = 10.0,
            min_vacuum_size: float = 10.0,
            height: float = 1.0,
            supercell_size: list[int] | None = None,
            calculator: BaseCalculator | None = None,
    ) -> None:
        """Initialize the SBE analyzer.

        Args:
            max_index (int): Maximum Miller index to consider when generating slabs.
            min_slab_size (float): Minimum slab thickness in Angstroms for slab generation.
            min_vacuum_size (float): Minimum vacuum size in Angstroms for slab generation.
            height (float): Height above the surface to identify surface atoms
            supercell_size (list[int]): Supercell replication factors used when creating the slab supercell for vacancy work.
            calculator (BaseCalculator | None, optional): The calculator object used to compute potential energies.
                                                                Defaults to `M3GNetCalculator`.
        """
        self.max_index = max_index
        self.min_slab_size = min_slab_size
        self.min_vacuum_size = min_vacuum_size
        self.height = height
        self.supercell_size = supercell_size if supercell_size is not None else [4, 4, 1]

        self.ase_adaptor = AseAtomsAdaptor()
        self._calculator = calculator

    def calculate(
            self,
            structure: Structure | Atoms,
            is_relaxed: bool = False
    ) -> dict[str, Any]:
        """Calculate the Surface Binding Energy for a given structure.

        Args:
            structure (Structure | Atoms): The undeformed structure to be transformed and analyzed.
            is_relaxed (bool, optional): Whether the input structure is already relaxed. Defaults to False.

        Returns:
            dict[str, Any]: A dictionary with the following keys
                - "miller_groups": A dictionary mapping Miller indices to their respective slab terminations and summary data.
                    Each Miller index maps to a dictionary with keys like "t_0", "t_1" for terminations, and summary keys like
                    "avg_surface_binding_energy" and "avg_surface_binding_energy_by_element".

        Raises:
            ValueError: If the calculator object does not have the 'energy' property implemented.
        """
        if "energy" not in self.calculator.AVAILABLE_PROPERTIES:
            raise ValueError("The calculator object must have the 'energy' property implemented.")

        if isinstance(structure, Atoms):
            structure = self.ase_adaptor.get_structure(structure)

        # (1) Relax the bulk structure
        if not is_relaxed:
            structure = self.calculator.relax(structure)["final_structure"]

        bulk_structure = structure.copy()
        bulk_energy_per_atom = self.calculator.calculate(bulk_structure)["energy"] / len(bulk_structure)

        # (2) Generate inequivalent slabs with Miller indices up to max_index and relax them to find the lowest energy surface configuration
        slabs = generate_all_slabs(
                structure=bulk_structure,
                max_index=self.max_index,
                min_slab_size=self.min_slab_size,
                min_vacuum_size=self.min_vacuum_size
        )

        miller_groups = {}
        best_gamma = float("inf")
        best_miller_index = None
        termination_counters = {}
        for slab in slabs:
            slab_area = slab.surface_area
            res_slab = self.calculator.relax(slab)
            relaxed_slab = res_slab["final_structure"]
            slab_energy = res_slab["energy"]
            gamma = (slab_energy - len(relaxed_slab) * bulk_energy_per_atom) / (2 * slab_area)

            miller_index = tuple(slab.miller_index)
            idx = termination_counters.get(miller_index, 0)
            miller_groups.setdefault(miller_index, {})[f"t_{idx}"] = {
                "unrelaxed_slab": slab.as_dict(),
                "relaxed_slab": relaxed_slab.as_dict(),
                "slab_area": slab_area,
                "slab_energy": slab_energy,
                "surface_energy": gamma,
            }
            termination_counters[miller_index] = idx + 1

            if gamma < best_gamma:
                best_gamma = gamma
                best_miller_index = miller_index

        if not miller_groups:
            raise ValueError("No slabs were generated for the given structure and parameters.")

        # (5) Calculate isolated atom energy
        isolated_atom_energies = {}
        for element in bulk_structure.elements:
            isolated_atom_energies[element.name] = self._calculate_isolated_atom_energy(element.name)

        # (3.2) Create supercells and calculate total energies for each slab termination
        for slab_termination in miller_groups[best_miller_index].values():
            supercell_slab = Structure.from_dict(slab_termination["relaxed_slab"]).make_supercell(self.supercell_size, in_place=False)
            supercell_slab_energy = self.calculator.calculate(supercell_slab)["energy"]  # E_s

            slab_termination["supercell_slab"] = supercell_slab.as_dict()
            slab_termination["supercell_slab_energy"] = supercell_slab_energy

            # (4) Identify surface atoms, create vacancies, and calculate E_s+v for each surface site
            site_finder = AdsorbateSiteFinder(slab=supercell_slab, height=self.height)

            surface_ids_by_element = {}
            for idx, site in enumerate(site_finder.slab):
                if site.properties.get("surface_properties") == "surface":
                    surface_ids_by_element.setdefault(site.species_string, []).append(idx)

            vacancy_groups = {}
            for slab_vacancy_element, surface_element_ids in surface_ids_by_element.items():
                for surface_element_id in surface_element_ids:
                    slab_vacancy = supercell_slab.copy()
                    slab_vacancy.remove_sites([surface_element_id])
                    slab_vacancy_energy = self.calculator.calculate(slab_vacancy)["energy"]
                    isolated_atom_energy = isolated_atom_energies[slab_vacancy_element]

                    vacancy_groups.setdefault(slab_vacancy_element, []).append({
                            "surface_element_id": surface_element_id,
                            "slab_vacancy": slab_vacancy.as_dict(),
                            "slab_vacancy_energy": slab_vacancy_energy,
                            "isolated_atom_energy": isolated_atom_energy,
                            "supercell_slab_energy": supercell_slab_energy,
                            "surface_binding_energy": isolated_atom_energy + slab_vacancy_energy - supercell_slab_energy,
                    })

            for el, entries in vacancy_groups.items():
                values = [entry["surface_binding_energy"] for entry in entries]
                slab_termination[f"{el}_avg_surface_binding_energy"] = float(np.mean(values))

            slab_termination["vacancy_groups"] = vacancy_groups

        avg_by_element = {}
        all_values = []
        for slab_termination in miller_groups[best_miller_index].values():
            for key, val in slab_termination.items():
                if key.endswith("_avg_surface_binding_energy"):
                    el = key[: -len("_avg_surface_binding_energy")]
                    valf = float(val)
                    if el in avg_by_element:
                        avg_by_element[el].append(valf)
                    else:
                        avg_by_element[el] = [valf]
                    all_values.append(valf)

        miller_groups[best_miller_index]["avg_surface_binding_energy_by_element"] = {
            el: float(np.mean(vals)) for el, vals in avg_by_element.items()
        }
        miller_groups[best_miller_index]["avg_surface_binding_energy"] = float(np.mean(all_values)) if all_values else None

        return miller_groups

    def _calculate_isolated_atom_energy(self, element: str) -> float:
        """Calculate the total energy of an isolated atom in a cubic cell.

        The isolated atom energy is computed by placing a single atom of the specified
        element in a large cubic cell (20 Angstroms per side) to minimize periodic interactions.

        Args:
            element (str): The chemical symbol of the element for which to calculate the isolated atom energy.

        Returns:
            float: Total energy returned by the calculator for the isolated atom configuration.
        """
        lattice = Lattice.cubic(20.0)
        isolated_atom = Structure(lattice, [element], [[0.0, 0.0, 0.0]])

        return self.calculator.calculate(structure=isolated_atom)["energy"]

    @property
    def calculator(self) -> BaseCalculator:
        """Returns the calculator instance used for energy calculations.

        Returns:
            BaseCalculator: The calculator object used for energy calculations.
        """
        if self._calculator is None:
            from materialsframework.calculators.m3gnet import M3GNetCalculator
            self._calculator = M3GNetCalculator()
        return self._calculator
