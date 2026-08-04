"""Integration tests for MatRISCalculator."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matris")

from materialsframework.calculators.matris import MatRISCalculator


@pytest.fixture(scope="module")
def calc() -> MatRISCalculator:
    """MatRISCalculator with the default matris_10m_oam foundation model."""
    return MatRISCalculator()


@pytest.mark.integration
def test_lazy_load_contract() -> None:
    """_calculator must be None until the calculator property is accessed."""
    c = MatRISCalculator()
    assert c._calculator is None


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES is exactly ['energy', 'forces', 'stress', 'magmoms']."""
    assert MatRISCalculator.AVAILABLE_PROPERTIES == ["energy", "forces", "stress", "magmoms"]


@pytest.mark.integration
def test_calculate_energy(calc: MatRISCalculator, bcc_fe) -> None:
    """calculate() returns a float energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "energy" in result
    assert isinstance(result["energy"], (float, np.floating))


@pytest.mark.integration
def test_calculate_forces_shape(calc: MatRISCalculator, bcc_fe) -> None:
    """calculate() returns forces with shape (n_atoms, 3)."""
    result = calc.calculate(bcc_fe)
    forces = np.array(result["forces"])
    assert forces.shape == (len(bcc_fe), 3)


@pytest.mark.integration
def test_calculate_stress(calc: MatRISCalculator, bcc_fe) -> None:
    """calculate() result includes a 'stress' entry."""
    result = calc.calculate(bcc_fe)
    assert "stress" in result


@pytest.mark.integration
def test_calculate_magmoms(calc: MatRISCalculator, bcc_fe) -> None:
    """calculate() returns non-null per-atom magnetic moments for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "magmoms" in result
    magmoms = np.array(result["magmoms"])
    assert magmoms.shape == (len(bcc_fe),)
    assert np.all(magmoms > 0)


@pytest.mark.integration
def test_relax_returns_structure(calc: MatRISCalculator, bcc_fe) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result
    assert "trajectory" in result
