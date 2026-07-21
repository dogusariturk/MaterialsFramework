"""Bond-based lattice parameter model for FCC, BCC, and HCP alloys.

Implements the method of Tandoc et al. (Materialia 2025) for predicting alloy lattice
parameters from pairwise nearest-neighbor bond lengths. Bond lengths are extracted from
ML potential relaxations of pure-element cells and binary intermetallic reference cells
(B2, L1_0, or D0_19, depending on the crystal structure), then combined into a
composition-weighted average bond length and converted to a lattice parameter.
"""

from __future__ import annotations

import csv
import math
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pymatgen.core import Lattice, Structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)


def fcc_cell(el: str, a: float) -> Structure:
    """Build a 4-atom conventional FCC cell (Fm-3m)."""
    coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    return Structure(Lattice.cubic(a), [el] * 4, coords)


def l10_cell(el_a: str, el_b: str, a_fcc: float) -> Structure:
    """Build a 2-atom L1_0 primitive cell (P4/mmm).

    Initial guess: a_tet = a_fcc / sqrt(2), c = a_fcc.
    """
    lattice = Lattice.tetragonal(a_fcc / SQRT2, a_fcc)
    return Structure(lattice, [el_a, el_b], [[0, 0, 0], [0.5, 0.5, 0.5]])


def bcc_cell(el: str, a: float) -> Structure:
    """Build a 2-atom conventional BCC cell (Im-3m)."""
    return Structure(Lattice.cubic(a), [el] * 2, [[0, 0, 0], [0.5, 0.5, 0.5]])


def b2_cell(el_a: str, el_b: str, a_bcc: float) -> Structure:
    """Build a 2-atom B2 (CsCl-type) cell (Pm-3m).

    Initial guess: a_bcc = average of two pure BCC lattice parameters.
    """
    return Structure(Lattice.cubic(a_bcc), [el_a, el_b], [[0, 0, 0], [0.5, 0.5, 0.5]])


def hcp_cell(el: str, a: float, c: float) -> Structure:
    """Build a 2-atom HCP cell (P6_3/mmc)."""
    lattice = Lattice.hexagonal(a, c)
    return Structure(lattice, [el] * 2, [[1 / 3, 2 / 3, 1 / 4], [2 / 3, 1 / 3, 3 / 4]])


def d019_cell(el_a: str, el_b: str, a: float, c: float) -> Structure:
    """Build an 8-atom D0_19 (Ni3Sn-type) cell (P6_3/mmc) for i-j bond extraction.

    el_a occupies the 6h majority sublattice (6 atoms), el_b occupies 2c (2 atoms).
    All 12 FNN of the minority el_b atoms are el_a atoms.
    """
    lattice = Lattice.hexagonal(a, c)
    coords_a = [
        [5 / 6, 2 / 3, 1 / 4],
        [1 / 3, 1 / 6, 1 / 4],
        [5 / 6, 1 / 6, 1 / 4],
        [1 / 6, 1 / 3, 3 / 4],
        [2 / 3, 5 / 6, 3 / 4],
        [1 / 6, 5 / 6, 3 / 4],
    ]
    coords_b = [
        [1 / 3, 2 / 3, 1 / 4],
        [2 / 3, 1 / 3, 3 / 4],
    ]
    return Structure(lattice, [el_a] * 6 + [el_b] * 2, coords_a + coords_b)


class BondLatticeParameter:
    """Bond-based lattice parameter model for FCC, BCC, or HCP alloy systems.

    Predicts lattice parameters from pairwise bond lengths obtained via ML potential
    relaxations, following Tandoc et al. (Materialia 2025).

    Two ways to create:

        model = BondLatticeParameter("fcc", ["Co", "Cr", "Fe", "Ni"])
        result = model.calculate()   # runs relaxations
        a = model.predict({"Co": 0.25, "Cr": 0.25, "Fe": 0.25, "Ni": 0.25})

        model = BondLatticeParameter.from_csv("fcc", "fcc_bond_model.csv")
        a = model.predict({"Co": 0.25, "Cr": 0.25, "Fe": 0.25, "Ni": 0.25})

    Args:
        structure: Crystal structure type: ``"fcc"``, ``"bcc"``, or ``"hcp"``.
        elements: Element symbols defining the alloy system.
        initial_a: Lattice parameter guesses per element. Falls back to
            built-in defaults for missing entries.
        initial_ca: c/a ratios per element for HCP. Falls back to built-in
            defaults (ideal c/a = 1.633 for non-native HCP elements).
            Ignored for FCC/BCC.
        calculator: Pre-configured calculator for relaxations.
    """

    _DEFAULT_A: dict[str, dict[str, float]] = {
        "fcc": {
            "Ag": 4.09,
            "Al": 4.05,
            "Au": 4.08,
            "Ca": 5.59,
            "Ce": 5.16,
            "Co": 3.54,
            "Cr": 3.68,
            "Cu": 3.61,
            "Fe": 3.59,
            "Ir": 3.84,
            "Mn": 3.63,
            "Mo": 3.90,
            "Nb": 4.23,
            "Ni": 3.52,
            "Pd": 3.89,
            "Pt": 3.92,
            "Rh": 3.80,
            "Ru": 3.82,
            "Sc": 4.54,
            "Si": 3.87,
            "Sn": 4.58,
            "Sr": 6.08,
            "Ta": 4.22,
            "Ti": 4.11,
            "V": 3.82,
            "W": 3.96,
            "Y": 5.08,
            "Zn": 3.94,
            "Zr": 4.53,
        },
        "bcc": {
            "Al": 3.24,
            "Ba": 5.02,
            "Co": 2.82,
            "Cr": 2.88,
            "Cs": 6.14,
            "Cu": 2.88,
            "Fe": 2.87,
            "Hf": 3.56,
            "K": 5.23,
            "Li": 3.49,
            "Mn": 2.91,
            "Mo": 3.15,
            "Na": 4.23,
            "Nb": 3.30,
            "Ni": 2.81,
            "Rb": 5.59,
            "Re": 3.19,
            "Ru": 3.15,
            "Sc": 3.63,
            "Ta": 3.30,
            "Ti": 3.31,
            "V": 3.02,
            "W": 3.16,
            "Zr": 3.58,
        },
        "hcp": {
            "Al": 2.86,
            "Be": 2.29,
            "Cd": 2.98,
            "Co": 2.51,
            "Cr": 2.72,
            "Cu": 2.56,
            "Fe": 2.58,
            "Hf": 3.20,
            "Mg": 3.21,
            "Mn": 2.73,
            "Mo": 2.77,
            "Nb": 2.94,
            "Ni": 2.49,
            "Os": 2.73,
            "Re": 2.76,
            "Ru": 2.71,
            "Sc": 3.31,
            "Ta": 2.94,
            "Ti": 2.95,
            "V": 2.62,
            "W": 2.77,
            "Y": 3.65,
            "Zn": 2.66,
            "Zr": 3.23,
        },
    }

    _DEFAULT_CA: dict[str, float] = {
        "Al": 1.633,
        "Be": 1.568,
        "Cd": 1.886,
        "Co": 1.622,
        "Cr": 1.633,
        "Cu": 1.633,
        "Fe": 1.603,
        "Hf": 1.582,
        "Mg": 1.624,
        "Mn": 1.633,
        "Mo": 1.633,
        "Nb": 1.633,
        "Ni": 1.633,
        "Os": 1.580,
        "Re": 1.615,
        "Ru": 1.584,
        "Sc": 1.594,
        "Ta": 1.633,
        "Ti": 1.587,
        "V": 1.633,
        "W": 1.633,
        "Y": 1.571,
        "Zn": 1.856,
        "Zr": 1.593,
    }

    def __init__(
        self,
        structure: Literal["fcc", "bcc", "hcp"],
        elements: Sequence[str],
        initial_a: dict[str, float] | None = None,
        initial_ca: dict[str, float] | None = None,
        calculator: BaseCalculator | None = None,
    ) -> None:
        """Initialize with crystal structure type, element system, and optional calculator."""
        self.structure = structure
        self.elements = list(elements)

        merged = dict(self._DEFAULT_A[structure])
        if initial_a:
            merged.update(initial_a)
        self.initial_a = {el: merged[el] for el in self.elements}

        if structure == "hcp":
            ca_merged = dict(self._DEFAULT_CA)
            if initial_ca:
                ca_merged.update(initial_ca)
            self.initial_ca = {el: ca_merged[el] for el in self.elements}
        else:
            self.initial_ca: dict[str, float] = {}

        self._calculator = calculator
        self.bonds: dict[tuple[str, str], float] = {}
        self.a_pure: dict[str, float] = {}

    @property
    def calculator(self) -> BaseCalculator:
        """Returns the calculator instance used for relaxations.

        If the calculator instance is not already initialized, this method creates a new
        ``M3GNetCalculator`` with settings suitable for bond-length extraction.

        Returns:
            BaseCalculator: The calculator object used for relaxations.
        """
        if self._calculator is None:
            from materialsframework.calculators.m3gnet import M3GNetCalculator

            self._calculator = M3GNetCalculator(fmax=0.01, optimizer="FIRE")

        self._calculator.fix_symmetry = True
        self._calculator.relax_cell = True

        return self._calculator

    @classmethod
    def from_csv(
        cls,
        structure: Literal["fcc", "bcc", "hcp"],
        path: str | Path = "bond_model.csv",
    ) -> BondLatticeParameter:
        """Create a prediction-only model from a symmetric bond-length matrix CSV.

        Format
            ,Co,Cr,Fe,...
            Co,2.5036,2.5512,2.4988,...
            Cr,2.5512,2.6021,...
            ...

        For FCC, pure lattice parameters are recovered as a_i = d_ii * sqrt(2).
        For BCC, pure lattice parameters are recovered as a_i = d_ii * 2 / sqrt(3).
        For HCP, pure lattice parameters are recovered as a_i = d_ii (exact for ideal c/a).
        """
        bonds: dict[tuple[str, str], float] = {}

        with Path(path).open(newline="") as f:
            reader = csv.reader(f)
            elements = next(reader)[1:]
            for row in reader:
                ei = row[0]
                for ej, val in zip(elements, row[1:], strict=True):
                    key = cls._bond_key(ei, ej)
                    if key not in bonds:
                        bonds[key] = float(val)

        if structure == "fcc":
            a_pure = {el: bonds[(el, el)] * SQRT2 for el in elements}
        elif structure == "hcp":
            a_pure = {el: bonds[(el, el)] for el in elements}
        else:
            a_pure = {el: bonds[(el, el)] * 2 / SQRT3 for el in elements}

        obj = object.__new__(cls)
        obj.structure = structure
        obj.elements = list(elements)
        obj.initial_a = {}
        obj.initial_ca = {}
        obj._calculator = None
        obj.bonds = bonds
        obj.a_pure = a_pure
        return obj

    @staticmethod
    def _bond_key(el_i: str, el_j: str) -> tuple[str, str]:
        """Return a canonical (sorted) pair key for the bond table."""
        return min(el_i, el_j), max(el_i, el_j)

    def _build_pure_cell(self, el: str, a: float) -> Structure:
        """Build a pure element cell (FCC, BCC, or HCP)."""
        if self.structure == "fcc":
            return fcc_cell(el, a)
        elif self.structure == "hcp":
            return hcp_cell(el, a, a * self.initial_ca[el])
        return bcc_cell(el, a)

    def _build_binary_cell(self, el_a: str, el_b: str, a: float) -> Structure:
        """Build a binary intermetallic cell (L1_0, B2, or D0_19)."""
        if self.structure == "fcc":
            return l10_cell(el_a, el_b, a)
        elif self.structure == "hcp":
            ca_avg = (self.initial_ca[el_a] + self.initial_ca[el_b]) / 2
            return d019_cell(el_a, el_b, a, a * ca_avg)
        return b2_cell(el_a, el_b, a)

    def _extract_pure_bond(self, lattice: Lattice) -> float:
        """Extract FNN bond length from a relaxed pure cell.

        FCC: d = a / sqrt(2).  BCC: d = a * sqrt(3) / 2.  HCP: d = sqrt(a^2/3 + c^2/4).
        """
        if self.structure == "fcc":
            return lattice.a / SQRT2
        elif self.structure == "hcp":
            return math.sqrt(lattice.a**2 / 3 + lattice.c**2 / 4)
        return lattice.a * SQRT3 / 2

    def _extract_binary_bond(self, lattice: Lattice) -> float:
        """Extract FNN bond length from a relaxed binary cell.

        L1_0 (tetragonal): d = sqrt(2a^2 + c^2) / 2.
        B2 (cubic):        d = a * sqrt(3) / 2.
        D0_19 (hexagonal):   d = sqrt(a^2/3 + c^2/4).
        """
        if self.structure == "fcc":
            return math.sqrt(2 * lattice.a**2 + lattice.c**2) / 2
        elif self.structure == "hcp":
            return math.sqrt(lattice.a**2 / 3 + lattice.c**2 / 4)
        return lattice.a * SQRT3 / 2

    def _d_to_a(self, d_bar: float) -> float:
        """Convert average bond length to lattice parameter.

        FCC: a = sqrt(2) * d.  BCC: a = 2 * d / sqrt(3).  HCP: a = d (ideal c/a).
        """
        if self.structure == "fcc":
            return SQRT2 * d_bar
        elif self.structure == "hcp":
            return d_bar
        return 2 * d_bar / SQRT3

    def calculate(self) -> dict[str, dict]:
        """Run all relaxations (pure + binary cells) and populate the bond table.

        Returns:
            dict[str, dict]: Dictionary with keys:
                - ``bonds``: Mapping of ``(el_i, el_j)`` pairs to FNN bond lengths.
                - ``a_pure``: Mapping of element symbols to relaxed pure lattice parameters.
        """
        self._relax_pure()
        self._relax_binaries()
        return {
            "bonds": dict(self.bonds),
            "a_pure": dict(self.a_pure),
        }

    def _relax_pure(self) -> None:
        """Relax all pure element cells and extract bond lengths."""
        for el in self.elements:
            result = self.calculator.relax(self._build_pure_cell(el, self.initial_a[el]))
            lat = result["final_structure"].lattice
            self.a_pure[el] = lat.a
            self.bonds[(el, el)] = self._extract_pure_bond(lat)

    def _relax_binaries(self) -> None:
        """Relax all binary intermetallic cells and extract bond lengths."""
        for el_a, el_b in combinations(self.elements, 2):
            a_avg = (self.initial_a[el_a] + self.initial_a[el_b]) / 2
            result = self.calculator.relax(self._build_binary_cell(el_a, el_b, a_avg))
            lat = result["final_structure"].lattice
            self.bonds[self._bond_key(el_a, el_b)] = self._extract_binary_bond(lat)

    def predict(self, composition: dict[str, float]) -> float:
        """Predict alloy lattice parameter from the bond table.

        FCC: a_bar = sqrt(2) * sum_ij x_i x_j d_ij.
        BCC: a_bar = (2/sqrt(3)) * sum_ij x_i x_j d_ij.
        HCP: a_bar = sum_ij x_i x_j d_ij (exact for ideal c/a).

        Args:
            composition: Mapping of element symbols to mole fractions.

        Returns:
            float: Predicted lattice parameter in Angstroms.

        Raises:
            ValueError: If ``calculate()`` has not been called and no bond data is available.
        """
        if not self.bonds:
            raise ValueError("No bond data available. Run calculate() first or load from CSV via from_csv().")
        d_bar = sum(composition[ei] * composition[ej] * self.bonds[self._bond_key(ei, ej)] for ei in composition for ej in composition)
        return self._d_to_a(d_bar)

    def vegard(self, composition: dict[str, float]) -> float:
        """Vegard's law estimate: a_bar = sum_i x_i a_i.

        Args:
            composition: Mapping of element symbols to mole fractions.

        Returns:
            float: Vegard's law lattice parameter in Angstroms.

        Raises:
            ValueError: If ``calculate()`` has not been called and no bond data is available.
        """
        if not self.a_pure:
            raise ValueError("No lattice parameter data available. Run calculate() first or load from CSV via from_csv().")
        return sum(composition[el] * self.a_pure[el] for el in composition)
