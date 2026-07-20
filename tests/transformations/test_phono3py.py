"""Tests for Phono3pyDisplacementTransformation."""

from __future__ import annotations

import pytest
from pymatgen.core import Structure

pytest.importorskip("phono3py")

from materialsframework.transformations.phono3py import (
    Phono3pyDisplacementTransformation,
)


def test_init() -> None:
    """Transformation initialises with no state attributes."""
    Phono3pyDisplacementTransformation()


@pytest.mark.integration
def test_apply_transformation_returns_phonon(bcc_fe) -> None:
    """apply_transformation returns a Phono3py object."""
    t = Phono3pyDisplacementTransformation()
    result = t.apply_transformation(bcc_fe)
    assert result["phonon"] is not None


@pytest.mark.integration
def test_apply_transformation_returns_supercells(bcc_fe) -> None:
    """apply_transformation returns non-empty supercell lists."""
    t = Phono3pyDisplacementTransformation()
    result = t.apply_transformation(bcc_fe)
    assert result["supercells_with_displacements"] is not None
    assert len(result["supercells_with_displacements"]) > 0
    for s in result["supercells_with_displacements"]:
        assert isinstance(s, Structure)


@pytest.mark.integration
def test_apply_transformation_returns_phonon_supercells(bcc_fe) -> None:
    """apply_transformation returns phonon supercells for 2nd-order force constants."""
    t = Phono3pyDisplacementTransformation()
    result = t.apply_transformation(bcc_fe)
    assert result["phonon_supercells_with_displacements"] is not None
    assert len(result["phonon_supercells_with_displacements"]) > 0
