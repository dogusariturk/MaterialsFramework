"""Shared session-scoped fixtures for the MaterialsFramework test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymatgen.core import Lattice, Structure

if TYPE_CHECKING:
    from ase import Atoms


@pytest.fixture(scope="session")
def bcc_fe() -> Structure:
    """Minimal 2-atom BCC Fe structure."""
    return Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture(scope="session")
def fcc_ni() -> Structure:
    """Minimal 4-atom FCC Ni structure."""
    return Structure(
        Lattice.cubic(3.52),
        ["Ni"] * 4,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )


@pytest.fixture(scope="session")
def primitive_bcc_fe() -> Structure:
    """1-atom primitive (rhombohedral) BCC Fe cell, not aligned with Cartesian axes."""
    a = 2.87
    lattice = Lattice([[-a / 2, a / 2, a / 2], [a / 2, -a / 2, a / 2], [a / 2, a / 2, -a / 2]])
    return Structure(lattice, ["Fe"], [[0, 0, 0]])


@pytest.fixture(scope="session")
def l10_feni() -> Structure:
    """2-atom L1_0 FeNi primitive cell (P4/mmm)."""
    return Structure(
        Lattice.tetragonal(2.53, 3.57),
        ["Fe", "Ni"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


@pytest.fixture(scope="session")
def ase_bcc_fe(bcc_fe) -> Atoms:
    """ASE Atoms version of the BCC Fe structure."""
    return bcc_fe.to_ase_atoms(msonable=False)


@pytest.fixture(scope="session")
def ase_l10_feni(l10_feni) -> Atoms:
    """ASE Atoms version of the L1_0 FeNi structure."""
    return l10_feni.to_ase_atoms(msonable=False)
