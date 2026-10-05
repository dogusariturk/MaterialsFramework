"""Integration tests for BAMCalculator.

BAMCalculator requires a model checkpoint path.
Set the BAM_MODEL env var to the path of a trained BAM model before running these tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("bam_torch")

from materialsframework.calculators.bam import BAMCalculator

MODEL_PATH = os.environ.get("BAM_MODEL", "")


@pytest.fixture(scope="module")
def calc() -> BAMCalculator:
    """BAMCalculator loaded from env-var model path, skipped if var is unset."""
    if not MODEL_PATH:
        pytest.skip(f"BAM model not found at {MODEL_PATH}")
    return BAMCalculator(model=MODEL_PATH)


@pytest.mark.integration
def test_lazy_load_contract() -> None:
    """_calculator must be None until the calculator property is accessed."""
    c = BAMCalculator(model="does-not-exist.pkl")
    assert c._calculator is None


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES is exactly ['energy', 'free_energy', 'forces', 'stress']."""
    assert BAMCalculator.AVAILABLE_PROPERTIES == ["energy", "free_energy", "forces", "stress"]


@pytest.mark.integration
def test_calculate_energy(calc: BAMCalculator, bcc_fe) -> None:
    """calculate() returns a float energy and free energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert isinstance(result["energy"], (float, np.floating))
    assert isinstance(result["free_energy"], (float, np.floating))


@pytest.mark.integration
def test_calculate_forces_and_stress_shape(calc: BAMCalculator, bcc_fe) -> None:
    """calculate() returns forces with shape (n_atoms, 3) and stress in Voigt form with shape (6,)."""
    result = calc.calculate(bcc_fe)
    assert np.array(result["forces"]).shape == (len(bcc_fe), 3)
    assert np.array(result["stress"]).shape == (6,)


@pytest.mark.integration
def test_relax_returns_structure(calc: BAMCalculator, bcc_fe) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result
    assert "trajectory" in result
