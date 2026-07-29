"""Tests for FormationEnergyTransformation."""

from __future__ import annotations

import warnings

import pytest
from pymatgen.core import Lattice, Structure

from materialsframework.transformations.formation_energy import (
    FormationEnergyTransformation,
)


def test_apply_transformation_populates_pure_structures(l10_feni) -> None:
    """apply_transformation generates one pure-element entry per composition element."""
    t = FormationEnergyTransformation()
    pure_structures = t.apply_transformation(l10_feni)
    assert len(pure_structures) == 2


def test_pure_structures_are_candidate_lists(l10_feni) -> None:
    """Each entry in pure_structures is a (str, list[Structure], int) tuple of candidate references."""
    t = FormationEnergyTransformation()
    pure_structures = t.apply_transformation(l10_feni)
    for element, candidates, num in pure_structures:
        assert isinstance(element, str)
        assert isinstance(candidates, list)
        assert len(candidates) >= 1
        for struct in candidates:
            assert isinstance(struct, Structure)
        assert num > 0


def test_known_ground_states_return_a_single_candidate(l10_feni) -> None:
    """Fe (BCC) and Ni (FCC) both have a tabulated ground state, so no guessing is needed."""
    t = FormationEnergyTransformation()
    pure_structures = t.apply_transformation(l10_feni)
    for _, candidates, _ in pure_structures:
        assert len(candidates) == 1


def test_apply_transformation_is_independent_across_calls(l10_feni) -> None:
    """Repeated calls to apply_transformation return independent, equally-sized results."""
    t = FormationEnergyTransformation()
    first = t.apply_transformation(l10_feni)
    second = t.apply_transformation(l10_feni)
    assert len(first) == len(second)
    assert first is not second


@pytest.mark.parametrize(
    ("element", "expected_sites"),
    [
        ("Cu", 1),
        ("W", 1),
        ("Mg", 2),
        ("Si", 2),
        ("Bi", 2),
        ("Sn", 2),
        ("Po", 1),
        ("Ne", 1),
    ],
)
def test_known_ground_state_prototypes_span_crystal_families(element, expected_sites) -> None:
    """Elements spanning distinct crystal families each resolve to their real, single-candidate ground state."""
    t = FormationEnergyTransformation()
    candidates = t._reference_candidates(element)
    assert len(candidates) == 1
    assert candidates[0].num_sites == expected_sites
    assert candidates[0].elements[0].symbol == element


@pytest.mark.parametrize("element", ["H", "N", "O", "F"])
def test_diatomic_gas_elements_return_isolated_dimer(element) -> None:
    """H, N, O, and F have no solid-state ground state; the reference is an isolated dimer in a vacuum box."""
    t = FormationEnergyTransformation()
    candidates = t._reference_candidates(element)
    assert len(candidates) == 1
    structure = candidates[0]
    assert structure.num_sites == 2
    assert structure.lattice.a == pytest.approx(20.0)


def test_noble_gas_atom_returns_isolated_atom_in_a_box() -> None:
    """He has no solid-state ground state at ambient pressure; the reference is a single isolated atom."""
    t = FormationEnergyTransformation()
    candidates = t._reference_candidates("He")
    assert len(candidates) == 1
    structure = candidates[0]
    assert structure.num_sites == 1
    assert structure.lattice.a == pytest.approx(20.0)


@pytest.mark.parametrize("element", ["Mn", "P", "S", "Ga", "B"])
def test_elements_requiring_an_atomic_basis_fall_back_to_guessed_candidates(element) -> None:
    """Mn/P/S/Ga/B ground states need a multi-atom basis ase.build.bulk can't construct from a formula alone."""
    t = FormationEnergyTransformation()
    candidates = t._reference_candidates(element)
    assert len(candidates) == 5


@pytest.mark.parametrize("element", ["Pm", "Am", "Cf"])
def test_untabulated_elements_fall_back_to_guessed_candidates(element) -> None:
    """Elements with no tabulated reference state at all also fall back to guessed candidates."""
    t = FormationEnergyTransformation()
    candidates = t._reference_candidates(element)
    assert len(candidates) == 5


def test_runtime_error_from_bulk_falls_back_to_guessed_candidates() -> None:
    """Pa's tetragonal ground state raises RuntimeError (not ValueError) from ase.build.bulk, which also falls back."""
    t = FormationEnergyTransformation()
    candidates = t._reference_candidates("Pa")
    assert len(candidates) == 5


def test_guessed_fallback_warns() -> None:
    """Falling back to a guessed reference (Mn) emits a warning naming the element."""
    t = FormationEnergyTransformation()
    with pytest.warns(UserWarning, match="Mn"):
        t._reference_candidates("Mn")


def test_known_ground_state_does_not_warn() -> None:
    """An element with a tabulated ground state (Fe) does not trigger a warning."""
    t = FormationEnergyTransformation()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t._reference_candidates("Fe")


def test_ternary_alloy_returns_one_entry_per_element() -> None:
    """A three-element structure produces exactly one pure-reference entry per distinct element."""
    structure = Structure(
        Lattice.cubic(3.6),
        ["Fe", "Ni", "Cr", "Fe"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )
    t = FormationEnergyTransformation()
    pure_structures = t.apply_transformation(structure)
    assert len(pure_structures) == 3
    counts = {element: num for element, _, num in pure_structures}
    assert counts == {"Fe": 2, "Ni": 1, "Cr": 1}


def test_disordered_structure_rounds_fractional_amounts() -> None:
    """Fractional site occupancies are rounded, not truncated, when counting atoms per element."""
    structure = Structure(
        Lattice.cubic(3.6),
        [{"Fe": 0.9, "Ni": 0.1}] * 3,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]],
    )
    t = FormationEnergyTransformation()
    pure_structures = t.apply_transformation(structure)
    counts = {element: num for element, _, num in pure_structures}
    assert counts == {"Fe": 3, "Ni": 0}
