"""Integration tests for DeePMDCalculator.

DeePMDCalculator requires a frozen model file (.pb or .pt).
Set the DEEPMD_MODEL env var to the model path before running these tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("deepmd")

from materialsframework.calculators.deepmd import DeePMDCalculator

MODEL_PATH = os.environ.get("DEEPMD_MODEL", "")
_MISSING = not MODEL_PATH


@pytest.fixture(scope="module")
def calc() -> DeePMDCalculator:
    """DeePMDCalculator loaded from env-var model path, skipped if var is unset."""
    if _MISSING:
        pytest.skip(f"DeePMD model not found at {MODEL_PATH}")
    return DeePMDCalculator(model=MODEL_PATH)


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES is exactly ['energy', 'free_energy', 'forces', 'virial', 'stress']."""
    assert DeePMDCalculator.AVAILABLE_PROPERTIES == ["energy", "free_energy", "forces", "virial", "stress"]


@pytest.mark.integration
def test_calculate_energy(calc: DeePMDCalculator, bcc_fe) -> None:
    """calculate() returns a float energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "energy" in result
    assert isinstance(result["energy"], (float, np.floating))


@pytest.mark.integration
def test_calculate_forces_shape(calc: DeePMDCalculator, bcc_fe) -> None:
    """calculate() returns forces with shape (n_atoms, 3)."""
    result = calc.calculate(bcc_fe)
    forces = np.array(result["forces"])
    assert forces.shape == (len(bcc_fe), 3)


@pytest.mark.integration
def test_relax_returns_structure(calc: DeePMDCalculator, bcc_fe) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result
    assert "trajectory" in result
