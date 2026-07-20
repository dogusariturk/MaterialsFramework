"""Tests for PhonopyDisplacementTransformation."""

from __future__ import annotations

import pytest
from pymatgen.core import Structure

pytest.importorskip("phonopy")

from materialsframework.transformations.phonopy import PhonopyDisplacementTransformation


def test_init() -> None:
    """Transformation initialises with no state attributes."""
    PhonopyDisplacementTransformation()


@pytest.mark.integration
def test_apply_transformation_returns_phonon(bcc_fe) -> None:
    """apply_transformation returns a Phonopy object."""
    t = PhonopyDisplacementTransformation()
    result = t.apply_transformation(bcc_fe)
    assert result["phonon"] is not None


@pytest.mark.integration
def test_apply_transformation_returns_displaced_structures(bcc_fe) -> None:
    """apply_transformation returns a non-empty list of displaced structures."""
    t = PhonopyDisplacementTransformation()
    result = t.apply_transformation(bcc_fe)
    assert result["displaced_structures"] is not None
    assert len(result["displaced_structures"]) > 0
    for s in result["displaced_structures"]:
        assert isinstance(s, Structure)


@pytest.mark.integration
def test_apply_transformation_returns_displacements(bcc_fe) -> None:
    """apply_transformation returns the displacement vectors."""
    t = PhonopyDisplacementTransformation()
    result = t.apply_transformation(bcc_fe)
    assert result["displacements"] is not None
