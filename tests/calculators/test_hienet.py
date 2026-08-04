"""Integration tests for HIENetCalculator.

HIENetCalculator requires a model checkpoint path.
Set the HIENET_MODEL env var to the model path before running these tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("hienet")

from materialsframework.calculators.hienet import HIENetCalculator

MODEL_PATH = os.environ.get("HIENET_MODEL", "")
_MISSING = not MODEL_PATH


@pytest.fixture(scope="module")
def calc() -> HIENetCalculator:
    """HIENetCalculator on CPU, skipped if the model file is not present."""
    if _MISSING:
        pytest.skip(f"HIENet model not found at {MODEL_PATH}")
    return HIENetCalculator(model=str(MODEL_PATH))


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES is exactly ['energy', 'free_energy', 'energies', 'forces', 'stress']."""
    assert HIENetCalculator.AVAILABLE_PROPERTIES == ["energy", "free_energy", "energies", "forces", "stress"]


@pytest.mark.integration
def test_calculate_energy(calc: HIENetCalculator, bcc_fe) -> None:
    """calculate() returns a float energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "energy" in result
    assert isinstance(result["energy"], (float, np.floating))


@pytest.mark.integration
def test_calculate_forces_shape(calc: HIENetCalculator, bcc_fe) -> None:
    """calculate() returns forces with shape (n_atoms, 3)."""
    result = calc.calculate(bcc_fe)
    forces = np.array(result["forces"])
    assert forces.shape == (len(bcc_fe), 3)


@pytest.mark.integration
def test_relax_returns_structure(calc: HIENetCalculator, bcc_fe) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result
    assert "trajectory" in result
