"""Tests for BondLatticeParameter and its cell-builder helpers."""

from __future__ import annotations

import pytest

from materialsframework.tools.bond_lattice_parameter import (
    BondLatticeParameter,
    bcc_cell,
    fcc_cell,
    hcp_cell,
)


def test_fcc_cell_has_four_atoms() -> None:
    """fcc_cell() returns a 4-atom conventional FCC cell."""
    cell = fcc_cell("Ni", 3.52)
    assert len(cell) == 4


def test_fcc_cell_single_species() -> None:
    """fcc_cell() contains only the specified element."""
    cell = fcc_cell("Ni", 3.52)
    assert all(s.species_string == "Ni" for s in cell)


def test_fcc_cell_lattice_parameter() -> None:
    """fcc_cell() lattice parameter matches the requested value."""
    cell = fcc_cell("Ni", 3.52)
    assert cell.lattice.a == pytest.approx(3.52)


def test_bcc_cell_has_two_atoms() -> None:
    """bcc_cell() returns a 2-atom conventional BCC cell."""
    cell = bcc_cell("Fe", 2.87)
    assert len(cell) == 2


def test_bcc_cell_single_species() -> None:
    """bcc_cell() contains only the specified element."""
    cell = bcc_cell("Fe", 2.87)
    assert all(s.species_string == "Fe" for s in cell)


def test_hcp_cell_has_two_atoms() -> None:
    """hcp_cell() returns a 2-atom HCP primitive cell."""
    cell = hcp_cell("Ti", 2.95, 4.686)
    assert len(cell) == 2


def test_hcp_cell_single_species() -> None:
    """hcp_cell() contains only the specified element."""
    cell = hcp_cell("Ti", 2.95, 4.686)
    assert all(s.species_string == "Ti" for s in cell)


def test_init_stores_structure_and_elements() -> None:
    """BondLatticeParameter stores structure type and element list."""
    model = BondLatticeParameter("bcc", ["Fe", "Co"])
    assert model.structure == "bcc"
    assert model.elements == ["Fe", "Co"]


def test_init_no_calculator_assigned() -> None:
    """BondLatticeParameter._calculator is None on construction."""
    model = BondLatticeParameter("bcc", ["Fe", "Co"])
    assert model._calculator is None


def test_init_bonds_and_a_pure_empty() -> None:
    """Bonds and a_pure dicts are empty before calculate()."""
    model = BondLatticeParameter("bcc", ["Fe", "Co"])
    assert model.bonds == {}
    assert model.a_pure == {}


def test_predict_raises_before_calculate() -> None:
    """predict() raises ValueError if no bond data is available."""
    model = BondLatticeParameter("bcc", ["Fe", "Co"])
    with pytest.raises(ValueError, match="No bond data available"):
        model.predict({"Fe": 0.5, "Co": 0.5})


def test_vegard_raises_before_calculate() -> None:
    """vegard() raises ValueError if no lattice parameter data is available."""
    model = BondLatticeParameter("bcc", ["Fe", "Co"])
    with pytest.raises(ValueError, match="No lattice parameter data available"):
        model.vegard({"Fe": 0.5, "Co": 0.5})


def test_initial_a_uses_defaults() -> None:
    """BondLatticeParameter pulls default lattice parameters from _DEFAULT_A."""
    model = BondLatticeParameter("bcc", ["Fe", "Co"])
    assert "Fe" in model.initial_a
    assert "Co" in model.initial_a
    assert model.initial_a["Fe"] == pytest.approx(2.87)


@pytest.fixture(scope="module")
def calc():
    """CHGNetCalculator on CPU, shared across all integration tests in this module."""
    pytest.importorskip("chgnet")
    from materialsframework.calculators.chgnet import CHGNetCalculator

    return CHGNetCalculator(device="cpu")


@pytest.fixture(scope="module")
def model(calc):
    """BondLatticeParameter for binary BCC Fe-Co with CHGNet calculator."""
    return BondLatticeParameter("bcc", ["Fe", "Co"], calculator=calc)


@pytest.fixture(scope="module")
def bonds_result(model):
    """Single calculate() result shared by all result-checking tests."""
    return model.calculate()


@pytest.mark.integration
def test_calculate_populates_bonds_and_a_pure(bonds_result) -> None:
    """calculate() fills bonds and a_pure dicts for a binary BCC system."""
    assert "bonds" in bonds_result
    assert "a_pure" in bonds_result
    assert ("Fe", "Fe") in bonds_result["bonds"]
    assert ("Co", "Co") in bonds_result["bonds"]
    assert "Fe" in bonds_result["a_pure"]
    assert "Co" in bonds_result["a_pure"]


@pytest.mark.integration
def test_predict_returns_reasonable_lattice_parameter(model, bonds_result) -> None:
    """predict() returns a lattice parameter in the physically expected range for Fe-Co BCC."""
    a = model.predict({"Fe": 0.5, "Co": 0.5})
    assert isinstance(a, float)
    assert 2.7 < a < 3.1


@pytest.mark.integration
def test_vegard_returns_reasonable_lattice_parameter(model, bonds_result) -> None:
    """vegard() returns a value consistent with the pure element parameters."""
    a_vegard = model.vegard({"Fe": 0.5, "Co": 0.5})
    assert isinstance(a_vegard, float)
    assert 2.7 < a_vegard < 3.1
