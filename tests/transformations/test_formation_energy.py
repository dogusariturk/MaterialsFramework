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
    """Each entry in pure_structures is a (list[Structure], int) tuple of candidate references."""
    t = FormationEnergyTransformation()
    pure_structures = t.apply_transformation(l10_feni)
    for candidates, num in pure_structures:
        assert isinstance(candidates, list)
        assert len(candidates) >= 1
        for struct in candidates:
            assert isinstance(struct, Structure)
        assert num > 0


def test_known_ground_states_return_a_single_candidate(l10_feni) -> None:
    """Fe (BCC) and Ni (FCC) both have a tabulated ground state, so no guessing is needed."""
    t = FormationEnergyTransformation()
    pure_structures = t.apply_transformation(l10_feni)
    for candidates, _ in pure_structures:
        assert len(candidates) == 1


def test_diatomic_gas_element_returns_isolated_dimer() -> None:
    """H has no solid-state ground state; the reference is an isolated H2 dimer in a vacuum box."""
    t = FormationEnergyTransformation()
    candidates = t._reference_candidates("H")
    assert len(candidates) == 1
    assert candidates[0].num_sites == 2


def test_element_requiring_atomic_basis_falls_back_to_guessed_candidates() -> None:
    """Mn's ground state (58-atom alpha-Mn) can't be built from ase.build.bulk.

    Several high-symmetry candidates are guessed instead.
    """
    t = FormationEnergyTransformation()
    candidates = t._reference_candidates("Mn")
    assert len(candidates) == 5  # FCC, BCC, HCP, diamond, simple cubic


def test_apply_transformation_is_independent_across_calls(l10_feni) -> None:
    """Repeated calls to apply_transformation return independent, equally-sized results."""
    t = FormationEnergyTransformation()
    first = t.apply_transformation(l10_feni)
    second = t.apply_transformation(l10_feni)
    assert len(first) == len(second)
    assert first is not second
