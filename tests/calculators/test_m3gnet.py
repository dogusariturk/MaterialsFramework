"""Integration tests for M3GNetCalculator."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matgl")

from materialsframework.calculators.m3gnet import M3GNetCalculator


@pytest.fixture(scope="module")
def calc() -> M3GNetCalculator:
    """M3GNetCalculator with default model."""
    return M3GNetCalculator()


@pytest.mark.integration
def test_lazy_load_contract() -> None:
    """_calculator must be None until the calculator property is accessed."""
    c = M3GNetCalculator()
    assert c._calculator is None


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES is exactly ['energy', 'forces', 'stress']."""
    assert M3GNetCalculator.AVAILABLE_PROPERTIES == ["energy", "forces", "stress"]


@pytest.mark.integration
def test_calculate_energy(calc: M3GNetCalculator, bcc_fe) -> None:
    """calculate() returns a negative float energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "energy" in result
    assert isinstance(result["energy"], (float, np.floating))
    assert result["energy"] < 0


@pytest.mark.integration
def test_calculate_forces_shape(calc: M3GNetCalculator, bcc_fe) -> None:
    """calculate() returns forces with shape (n_atoms, 3)."""
    result = calc.calculate(bcc_fe)
    forces = np.array(result["forces"])
    assert forces.shape == (len(bcc_fe), 3)


@pytest.mark.integration
def test_calculate_stress(calc: M3GNetCalculator, bcc_fe) -> None:
    """calculate() result includes a 'stress' entry."""
    result = calc.calculate(bcc_fe)
    assert "stress" in result


@pytest.mark.integration
def test_relax_returns_structure(calc: M3GNetCalculator, bcc_fe) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result
    assert "trajectory" in result
