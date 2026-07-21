"""Integration tests for AllegroCalculator.

AllegroCalculator requires a trained model checkpoint.
Set the ALLEGRO_MODEL env var to the compiled model path before running these tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("allegro")

from materialsframework.calculators.allegro import AllegroCalculator

MODEL_PATH = os.environ.get("ALLEGRO_MODEL", "")
_MISSING = not MODEL_PATH


@pytest.fixture(scope="module")
def calc() -> AllegroCalculator:
    """AllegroCalculator loaded from env-var model path, skipped if var is unset."""
    if _MISSING:
        pytest.skip("ALLEGRO_MODEL env var not set")
    return AllegroCalculator(model=MODEL_PATH)


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES includes 'energy' and 'forces'."""
    assert "energy" in AllegroCalculator.AVAILABLE_PROPERTIES
    assert "forces" in AllegroCalculator.AVAILABLE_PROPERTIES


@pytest.mark.integration
def test_calculate_energy(calc: AllegroCalculator, bcc_fe) -> None:
    """calculate() returns a float energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "energy" in result
    assert isinstance(result["energy"], (float, np.floating))


@pytest.mark.integration
def test_calculate_forces_shape(calc: AllegroCalculator, bcc_fe) -> None:
    """calculate() returns forces with shape (n_atoms, 3)."""
    result = calc.calculate(bcc_fe)
    forces = np.array(result["forces"])
    assert forces.shape == (len(bcc_fe), 3)


@pytest.mark.integration
def test_relax_returns_structure(calc: AllegroCalculator, bcc_fe) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result
    assert "trajectory" in result
