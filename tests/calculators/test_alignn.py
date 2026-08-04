"""Integration tests for AlignnCalculator."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("alignn")

from materialsframework.calculators.alignn import AlignnCalculator


@pytest.fixture(scope="module")
def calc() -> AlignnCalculator:
    """AlignnCalculator with default model."""
    return AlignnCalculator()


@pytest.mark.integration
def test_lazy_load_contract() -> None:
    """_calculator must be None until the calculator property is accessed."""
    c = AlignnCalculator()
    assert c._calculator is None


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES is exactly ['energy', 'forces', 'stress']."""
    assert AlignnCalculator.AVAILABLE_PROPERTIES == ["energy", "forces", "stress"]


@pytest.mark.integration
def test_calculate_energy(calc: AlignnCalculator, bcc_fe) -> None:
    """calculate() returns a float energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "energy" in result
    assert isinstance(result["energy"], (float, np.floating))


@pytest.mark.integration
def test_calculate_forces_shape(calc: AlignnCalculator, bcc_fe) -> None:
    """calculate() returns forces with shape (n_atoms, 3)."""
    result = calc.calculate(bcc_fe)
    forces = np.array(result["forces"])
    assert forces.shape == (len(bcc_fe), 3)


@pytest.mark.integration
def test_relax_returns_structure(calc: AlignnCalculator, bcc_fe) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result
    assert "trajectory" in result
