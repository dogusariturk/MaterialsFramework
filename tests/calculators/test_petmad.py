"""Integration tests for PetMadCalculator."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("upet")

from materialsframework.calculators.petmad import PetMadCalculator


@pytest.fixture(scope="module")
def calc() -> PetMadCalculator:
    """PetMadCalculator with default model."""
    return PetMadCalculator()


@pytest.mark.integration
def test_lazy_load_contract() -> None:
    """_calculator must be None until the calculator property is accessed."""
    c = PetMadCalculator()
    assert c._calculator is None


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES includes 'energy' and 'forces'."""
    assert "energy" in PetMadCalculator.AVAILABLE_PROPERTIES
    assert "forces" in PetMadCalculator.AVAILABLE_PROPERTIES


@pytest.mark.integration
def test_calculate_energy(calc: PetMadCalculator, bcc_fe) -> None:
    """calculate() returns a negative float energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "energy" in result
    assert isinstance(result["energy"], (float, np.floating))
    assert result["energy"] < 0


@pytest.mark.integration
def test_calculate_forces_shape(calc: PetMadCalculator, bcc_fe) -> None:
    """calculate() returns forces with shape (n_atoms, 3)."""
    result = calc.calculate(bcc_fe)
    forces = np.array(result["forces"])
    assert forces.shape == (len(bcc_fe), 3)


@pytest.mark.integration
def test_relax_returns_structure(calc: PetMadCalculator, bcc_fe) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result
    assert "trajectory" in result
