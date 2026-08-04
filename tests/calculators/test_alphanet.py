"""Integration tests for AlphaNetCalculator.

AlphaNetCalculator requires a model checkpoint path and a config file.
Set the environment variables ALPHANET_MODEL and ALPHANET_CONFIG to the
respective paths before running these tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("alphanet")

from materialsframework.calculators.alphanet import AlphaNetCalculator

MODEL_PATH = os.environ.get("ALPHANET_MODEL", "")
CONFIG_PATH = os.environ.get("ALPHANET_CONFIG", "")
_MISSING = not MODEL_PATH or not CONFIG_PATH


@pytest.fixture(scope="module")
def calc() -> AlphaNetCalculator:
    """AlphaNetCalculator loaded from env-var paths, skipped if vars are unset."""
    if _MISSING:
        pytest.skip("ALPHANET_MODEL and ALPHANET_CONFIG env vars not set")
    return AlphaNetCalculator(model=MODEL_PATH, config=CONFIG_PATH)


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES is exactly ['energy', 'free_energy', 'forces', 'stress']."""
    assert AlphaNetCalculator.AVAILABLE_PROPERTIES == ["energy", "free_energy", "forces", "stress"]


@pytest.mark.integration
def test_calculate_energy(calc: AlphaNetCalculator, bcc_fe) -> None:
    """calculate() returns a float energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "energy" in result
    assert isinstance(result["energy"], (float, np.floating))


@pytest.mark.integration
def test_calculate_forces_shape(calc: AlphaNetCalculator, bcc_fe) -> None:
    """calculate() returns forces with shape (n_atoms, 3)."""
    result = calc.calculate(bcc_fe)
    forces = np.array(result["forces"])
    assert forces.shape == (len(bcc_fe), 3)


@pytest.mark.integration
def test_relax_returns_structure(calc: AlphaNetCalculator, bcc_fe) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result
    assert "trajectory" in result
