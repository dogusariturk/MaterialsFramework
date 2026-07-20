"""Tests for FormationEnergyTransformation."""

from __future__ import annotations

from pymatgen.core import Structure

from materialsframework.transformations.formation_energy import (
    FormationEnergyTransformation,
)


def test_apply_transformation_populates_pure_structures(l10_feni) -> None:
    """apply_transformation generates one pure-element entry per composition element."""
    t = FormationEnergyTransformation()
    pure_structures = t.apply_transformation(l10_feni)
    assert len(pure_structures) == 2  # Fe and Ni


def test_pure_structures_are_candidate_lists(l10_feni) -> None:
    """Each entry in pure_structures is a (list[Structure], int) tuple with FCC/BCC/HCP candidates."""
    t = FormationEnergyTransformation()
    pure_structures = t.apply_transformation(l10_feni)
    for candidates, num in pure_structures:
        assert isinstance(candidates, list)
        assert len(candidates) == 3  # FCC, BCC, HCP
        for struct in candidates:
            assert isinstance(struct, Structure)
        assert num > 0


def test_apply_transformation_is_independent_across_calls(l10_feni) -> None:
    """Repeated calls to apply_transformation return independent, equally-sized results."""
    t = FormationEnergyTransformation()
    first = t.apply_transformation(l10_feni)
    second = t.apply_transformation(l10_feni)
    assert len(first) == len(second)
    assert first is not second
