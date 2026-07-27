"""Tests for the elastic module utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from materialsframework.tools.elastic import (
    get_cart_deformed_cell,
    get_cij_order,
    get_elementary_deformations,
    get_lattice_type,
    get_pressure,
    get_strain,
    hexagonal,
    orthorombic,
    regular,
    tetragonal,
)
from materialsframework.utils import to_atoms


def test_regular_shape() -> None:
    """regular() returns a (6, 3) matrix for cubic symmetry."""
    mat = regular([1, 0, 0, 0, 0, 0])
    assert mat.shape == (6, 3)


def test_tetragonal_shape() -> None:
    """tetragonal() returns a (6, 6) matrix."""
    mat = tetragonal([1, 0, 0, 0, 0, 0])
    assert mat.shape == (6, 6)


def test_orthorombic_shape() -> None:
    """orthorombic() returns a (6, 9) matrix."""
    mat = orthorombic([1, 0, 0, 0, 0, 0])
    assert mat.shape == (6, 9)


def test_hexagonal_shape() -> None:
    """hexagonal() returns a (6, 5) matrix."""
    mat = hexagonal([1, 0, 0, 0, 0, 0])
    assert mat.shape == (6, 5)


def test_get_pressure_hydrostatic() -> None:
    """get_pressure returns the negative mean of the first three stress components."""
    stress = np.array([-2.0, -2.0, -2.0, 0.0, 0.0, 0.0])
    assert get_pressure(stress) == pytest.approx(2.0)


def test_get_pressure_zero() -> None:
    """get_pressure returns 0 for a zero-stress tensor."""
    assert get_pressure(np.zeros(6)) == pytest.approx(0.0)


def test_get_lattice_type_cubic(bcc_fe) -> None:
    """BCC Fe is classified as cubic (type 7)."""
    atoms = to_atoms(bcc_fe)
    ltype, brav, sg_name, sg_nr = get_lattice_type(atoms)
    assert ltype == 7
    assert brav == "Cubic"
    assert sg_name is None
    assert sg_nr is None


def test_get_cij_order_cubic(bcc_fe) -> None:
    """Cubic BCC Fe returns 3 independent elastic constants."""
    atoms = to_atoms(bcc_fe)
    order = get_cij_order(atoms)
    assert "C_11" in order
    assert "C_12" in order
    assert "C_44" in order
    assert len(order) == 3


def test_get_cart_deformed_cell_changes_cell(bcc_fe) -> None:
    """Deformed cell differs from the original along the deformation axis."""
    atoms = to_atoms(bcc_fe)
    deformed = get_cart_deformed_cell(atoms, axis=0, size=1.0)
    assert not np.allclose(deformed.get_cell(), atoms.get_cell())


def test_get_cart_deformed_cell_preserves_sites(bcc_fe) -> None:
    """Deformation does not change the number of atoms."""
    atoms = to_atoms(bcc_fe)
    deformed = get_cart_deformed_cell(atoms, axis=0, size=1.0)
    assert len(deformed) == len(atoms)


def test_get_strain_returns_six_components(bcc_fe) -> None:
    """get_strain returns a 6-element Voigt strain vector."""
    atoms = to_atoms(bcc_fe)
    deformed = get_cart_deformed_cell(atoms, axis=0, size=1.0)
    strain = get_strain(deformed, refcell=atoms)
    assert strain.shape == (6,)


def test_get_strain_identity_is_zero(bcc_fe) -> None:
    """Strain of a structure relative to itself is the zero vector."""
    atoms = to_atoms(bcc_fe)
    strain = get_strain(atoms, refcell=atoms)
    assert np.allclose(strain, 0.0, atol=1e-12)


def test_get_elementary_deformations_cubic_count(bcc_fe) -> None:
    """Cubic BCC Fe with n=3 produces 6 elementary deformations (3 axial + 3 shear)."""
    atoms = to_atoms(bcc_fe)
    deformations = get_elementary_deformations(atoms, n=3, d=1.0)
    assert len(deformations) == 6


def test_get_elementary_deformations_returns_atoms(bcc_fe) -> None:
    """All elementary deformations are ASE Atoms objects."""
    from ase import Atoms

    atoms = to_atoms(bcc_fe)
    deformations = get_elementary_deformations(atoms, n=3, d=1.0)
    for d in deformations:
        assert isinstance(d, Atoms)
