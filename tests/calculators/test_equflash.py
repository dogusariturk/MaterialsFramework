"""Integration tests for EquFlashCalculator.

EquFlashCalculator requires a downloaded checkpoint file.
Set the EQUFLASH_MODEL env var to the checkpoint path before running these tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("GGNN")

from materialsframework.calculators.equflash import EquFlashCalculator

MODEL_PATH = os.environ.get("EQUFLASH_MODEL", "")
_MISSING = not MODEL_PATH


@pytest.fixture(scope="module")
def calc() -> EquFlashCalculator:
    """EquFlashCalculator loaded from env-var checkpoint path, skipped if var is unset."""
    if _MISSING:
        pytest.skip(f"EquFlash model not found at {MODEL_PATH}")
    return EquFlashCalculator(model=MODEL_PATH)


@pytest.mark.integration
def test_lazy_load_contract() -> None:
    """_calculator must be None until the calculator property is accessed."""
    c = EquFlashCalculator()
    assert c._calculator is None


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES is exactly ['energy', 'forces', 'stress']."""
    assert EquFlashCalculator.AVAILABLE_PROPERTIES == ["energy", "forces", "stress"]


@pytest.mark.integration
@pytest.mark.slow
def test_calculate_energy(calc: EquFlashCalculator, bcc_fe) -> None:
    """calculate() returns a float energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "energy" in result
    assert isinstance(result["energy"], (float, np.floating))


@pytest.mark.integration
@pytest.mark.slow
def test_calculate_forces_shape(calc: EquFlashCalculator, bcc_fe) -> None:
    """calculate() returns forces with shape (n_atoms, 3)."""
    result = calc.calculate(bcc_fe)
    forces = np.array(result["forces"])
    assert forces.shape == (len(bcc_fe), 3)


@pytest.mark.integration
@pytest.mark.slow
def test_relax_returns_structure(calc: EquFlashCalculator, bcc_fe) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result
    assert "trajectory" in result
