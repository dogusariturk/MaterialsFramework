"""Tests for HSolubilityTransformation."""

from __future__ import annotations

import pytest

from materialsframework.transformations.h_solubility import HSolubilityTransformation
from materialsframework.utils import to_atoms


def test_init() -> None:
    """Transformation initializes expected default state."""
    t = HSolubilityTransformation()
    assert t._sqs_gen is None


def test_apply_transformation_returns_octahedral_and_tetrahedral(bcc_fe) -> None:
    """apply_transformation returns one structure per default site type."""
    t = HSolubilityTransformation()
    result = t.apply_transformation(bcc_fe)
    structures = result["structures"]

    assert set(structures) == {"octahedral", "tetrahedral"}
    assert len(structures["octahedral"]) == 1
    assert len(structures["tetrahedral"]) == 1

    for generated in structures["octahedral"] + structures["tetrahedral"]:
        assert generated.num_sites == bcc_fe.num_sites + 1
        assert generated.composition["H"] == 1


def test_apply_transformation_accepts_ase_atoms(bcc_fe) -> None:
    """apply_transformation accepts ASE Atoms inputs."""
    ase_bcc_fe = to_atoms(bcc_fe)
    t = HSolubilityTransformation()
    result = t.apply_transformation(ase_bcc_fe, site_types=("octahedral",))

    assert len(result["structures"]["octahedral"]) == 1


def test_apply_transformation_validates_site_types(bcc_fe) -> None:
    """Unknown site labels raise ValueError."""
    t = HSolubilityTransformation()

    with pytest.raises(ValueError, match="Invalid site type"):
        t.apply_transformation(bcc_fe, site_types=("bridge",))
