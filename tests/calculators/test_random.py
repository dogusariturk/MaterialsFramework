"""Tests for RandomCalculator (no ML extras required)."""

from __future__ import annotations

from materialsframework.calculators.random import RandomCalculator


def test_available_properties_exact() -> None:
    """AVAILABLE_PROPERTIES is exactly ['energy', 'forces']."""
    assert RandomCalculator.AVAILABLE_PROPERTIES == ["energy", "forces"]


def test_calculate_returns_required_keys(bcc_fe) -> None:
    """calculate() returns a dict containing all AVAILABLE_PROPERTIES."""
    result = RandomCalculator().calculate(bcc_fe)
    for key in RandomCalculator.AVAILABLE_PROPERTIES:
        assert key in result


def test_calculate_energy_is_float(bcc_fe) -> None:
    """Energy returned by calculate() is a float."""
    result = RandomCalculator().calculate(bcc_fe)
    assert isinstance(result["energy"], float)


def test_calculate_accepts_ase_atoms(ase_bcc_fe) -> None:
    """calculate() accepts ase.Atoms as well as pymatgen Structure."""
    result = RandomCalculator().calculate(ase_bcc_fe)
    assert "energy" in result
    assert "forces" in result


def test_relax_accepts_ase_atoms(ase_bcc_fe) -> None:
    """relax() accepts ase.Atoms and returns 'final_structure'."""
    result = RandomCalculator().relax(ase_bcc_fe)
    assert "final_structure" in result
