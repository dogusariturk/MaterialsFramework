"""Integration tests for TACECalculator."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tace")

from materialsframework.calculators.tace import TACECalculator


@pytest.fixture(scope="module")
def calc() -> TACECalculator:
    """TACECalculator with the default TACE-OAM-L foundation model."""
    return TACECalculator()


@pytest.mark.integration
def test_lazy_load_contract() -> None:
    """_calculator must be None until the calculator property is accessed."""
    c = TACECalculator()
    assert c._calculator is None


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES is exactly ['energy', 'free_energy', 'forces', 'stress']."""
    assert TACECalculator.AVAILABLE_PROPERTIES == ["energy", "free_energy", "forces", "stress"]


@pytest.mark.integration
def test_calculate_energy(calc: TACECalculator, bcc_fe) -> None:
    """calculate() returns a float energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "energy" in result
    assert isinstance(result["energy"], (float, np.floating))


@pytest.mark.integration
def test_calculate_forces_shape(calc: TACECalculator, bcc_fe) -> None:
    """calculate() returns forces with shape (n_atoms, 3)."""
    result = calc.calculate(bcc_fe)
    forces = np.array(result["forces"])
    assert forces.shape == (len(bcc_fe), 3)


@pytest.mark.integration
def test_calculate_stress(calc: TACECalculator, bcc_fe) -> None:
    """calculate() result includes a 'stress' entry."""
    result = calc.calculate(bcc_fe)
    assert "stress" in result


@pytest.mark.integration
def test_relax_returns_structure(calc: TACECalculator, bcc_fe) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result
    assert "trajectory" in result
