"""Tests for BaseCalculator using RandomCalculator (no ML extras required)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pymatgen.core import Structure

from materialsframework.calculators.random import RandomCalculator


@pytest.fixture
def calc() -> RandomCalculator:
    """RandomCalculator instance for testing BaseCalculator behavior."""
    return RandomCalculator()


def test_available_properties_defined() -> None:
    """AVAILABLE_PROPERTIES must be a non-empty list."""
    assert isinstance(RandomCalculator.AVAILABLE_PROPERTIES, list)
    assert len(RandomCalculator.AVAILABLE_PROPERTIES) > 0


def test_calculate_returns_required_keys(calc: RandomCalculator, bcc_fe: Structure) -> None:
    """calculate() returns a dict containing all AVAILABLE_PROPERTIES."""
    result = calc.calculate(bcc_fe)
    for key in RandomCalculator.AVAILABLE_PROPERTIES:
        assert key in result


def test_calculate_energy_is_float(calc: RandomCalculator, bcc_fe: Structure) -> None:
    """Energy returned by calculate() is a float."""
    result = calc.calculate(bcc_fe)
    assert isinstance(result["energy"], float)


def test_relax_returns_final_structure(calc: RandomCalculator, bcc_fe: Structure) -> None:
    """relax() returns a dict with 'final_structure' key."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result


def test_default_fmax() -> None:
    """Default fmax is 0.1 eV/A."""
    calc = RandomCalculator()
    assert calc.fmax == pytest.approx(0.1)


def test_custom_fmax() -> None:
    """Custom fmax is stored on the instance."""
    calc = RandomCalculator(fmax=0.05)
    assert calc.fmax == pytest.approx(0.05)


def test_calculate_accepts_ase_atoms(ase_bcc_fe) -> None:
    """calculate() accepts ase.Atoms as well as pymatgen Structure."""
    result = RandomCalculator().calculate(ase_bcc_fe)
    for key in RandomCalculator.AVAILABLE_PROPERTIES:
        assert key in result


def test_relax_accepts_ase_atoms(ase_bcc_fe) -> None:
    """relax() accepts ase.Atoms and returns 'final_structure'."""
    result = RandomCalculator().relax(ase_bcc_fe)
    assert "final_structure" in result
