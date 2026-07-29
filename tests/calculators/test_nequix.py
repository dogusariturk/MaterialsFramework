"""Integration tests for NequixCalculator."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("nequix")

from materialsframework.calculators.nequix import NequixCalculator


@pytest.fixture(scope="module")
def calc() -> NequixCalculator:
    """NequixCalculator with default model."""
    return NequixCalculator()


@pytest.mark.integration
def test_lazy_load_contract() -> None:
    """_calculator must be None until the calculator property is accessed."""
    c = NequixCalculator()
    assert c._calculator is None


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES includes 'energy' and 'forces'."""
    assert "energy" in NequixCalculator.AVAILABLE_PROPERTIES
    assert "forces" in NequixCalculator.AVAILABLE_PROPERTIES


@pytest.mark.integration
def test_calculate_energy(calc: NequixCalculator, bcc_fe) -> None:
    """calculate() returns a negative float energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "energy" in result
    assert isinstance(result["energy"], (float, np.floating))
    assert result["energy"] < 0


@pytest.mark.integration
def test_calculate_forces_shape(calc: NequixCalculator, bcc_fe) -> None:
    """calculate() returns forces with shape (n_atoms, 3)."""
    result = calc.calculate(bcc_fe)
    forces = np.array(result["forces"])
    assert forces.shape == (len(bcc_fe), 3)


@pytest.mark.integration
def test_relax_returns_structure(calc: NequixCalculator, bcc_fe) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result
    assert "trajectory" in result
