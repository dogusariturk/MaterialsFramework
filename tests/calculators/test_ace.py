"""Integration tests for ACECalculator.

ACECalculator requires a model path.
Set the ACE_MODEL env var to the path of a Si ACE model before running these tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("ace_jax")

from materialsframework.calculators.ace import ACECalculator

MODEL_PATH = os.environ.get("ACE_MODEL", "")


@pytest.fixture(scope="module")
def calc() -> ACECalculator:
    """ACECalculator loaded from env-var model path, skipped if var is unset."""
    if not MODEL_PATH:
        pytest.skip(f"ACE model not found at {MODEL_PATH}")
    return ACECalculator(model=MODEL_PATH)


@pytest.mark.integration
def test_lazy_load_contract() -> None:
    """_calculator must be None until the calculator property is accessed."""
    c = ACECalculator(model="does-not-exist.npz")
    assert c._calculator is None


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES is exactly ['energy', 'free_energy', 'forces', 'stress']."""
    assert ACECalculator.AVAILABLE_PROPERTIES == ["energy", "free_energy", "forces", "stress"]


@pytest.mark.integration
def test_calculate_energy(calc: ACECalculator, diamond_si) -> None:
    """calculate() returns a float energy and free energy for diamond Si."""
    result = calc.calculate(diamond_si)
    assert isinstance(result["energy"], (float, np.floating))
    assert isinstance(result["free_energy"], (float, np.floating))


@pytest.mark.integration
def test_calculate_forces_and_stress_shape(calc: ACECalculator, diamond_si) -> None:
    """calculate() returns forces with shape (n_atoms, 3) and stress in Voigt form with shape (6,)."""
    result = calc.calculate(diamond_si)
    assert np.array(result["forces"]).shape == (len(diamond_si), 3)
    assert np.array(result["stress"]).shape == (6,)


@pytest.mark.integration
def test_relax_returns_structure(calc: ACECalculator, diamond_si) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(diamond_si)
    assert "final_structure" in result
    assert "trajectory" in result
