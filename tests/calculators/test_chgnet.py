"""Integration tests for CHGNetCalculator."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("chgnet")

from materialsframework.calculators.chgnet import CHGNetCalculator


@pytest.fixture(scope="module")
def calc() -> CHGNetCalculator:
    """CHGNetCalculator with default model."""
    return CHGNetCalculator()


@pytest.mark.integration
def test_lazy_load_contract() -> None:
    """_calculator must be None until the calculator property is accessed."""
    c = CHGNetCalculator()
    assert c._calculator is None


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES is exactly ['energy', 'forces', 'stress', 'magmoms']."""
    assert CHGNetCalculator.AVAILABLE_PROPERTIES == ["energy", "forces", "stress", "magmoms"]


@pytest.mark.integration
def test_calculate_energy(calc: CHGNetCalculator, bcc_fe) -> None:
    """calculate() returns a negative float energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "energy" in result
    assert isinstance(result["energy"], (float, np.floating))
    assert result["energy"] < 0


@pytest.mark.integration
def test_calculate_forces_shape(calc: CHGNetCalculator, bcc_fe) -> None:
    """calculate() returns forces with shape (n_atoms, 3)."""
    result = calc.calculate(bcc_fe)
    forces = np.array(result["forces"])
    assert forces.shape == (len(bcc_fe), 3)


@pytest.mark.integration
def test_calculate_stress(calc: CHGNetCalculator, bcc_fe) -> None:
    """calculate() result includes a 'stress' entry."""
    result = calc.calculate(bcc_fe)
    assert "stress" in result


@pytest.mark.integration
def test_relax_returns_structure(calc: CHGNetCalculator, bcc_fe) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result
    assert "trajectory" in result
