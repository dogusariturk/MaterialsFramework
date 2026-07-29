"""Integration tests for EqV2Calculator.

EqV2Calculator downloads its checkpoint from the gated "facebook/OMAT24" Hugging Face repo.
Set the HF_TOKEN env var to an access token authorized for that repo before running these tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("fairchem.core")

from materialsframework.calculators.eqv2 import EqV2Calculator

_MISSING = not os.environ.get("HF_TOKEN")


@pytest.fixture(scope="module")
def calc() -> EqV2Calculator:
    """EqV2Calculator with default model, skipped if HF_TOKEN is unset."""
    if _MISSING:
        pytest.skip("HF_TOKEN not set; cannot download the gated facebook/OMAT24 checkpoint")
    return EqV2Calculator()


@pytest.mark.integration
def test_lazy_load_contract() -> None:
    """_calculator must be None until the calculator property is accessed."""
    c = EqV2Calculator()
    assert c._calculator is None


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES includes 'energy' and 'forces'."""
    assert "energy" in EqV2Calculator.AVAILABLE_PROPERTIES
    assert "forces" in EqV2Calculator.AVAILABLE_PROPERTIES


@pytest.mark.integration
def test_calculate_energy(calc: EqV2Calculator, bcc_fe) -> None:
    """calculate() returns a negative float energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "energy" in result
    assert isinstance(result["energy"], (float, np.floating))
    assert result["energy"] < 0


@pytest.mark.integration
def test_calculate_forces_shape(calc: EqV2Calculator, bcc_fe) -> None:
    """calculate() returns forces with shape (n_atoms, 3)."""
    result = calc.calculate(bcc_fe)
    forces = np.array(result["forces"])
    assert forces.shape == (len(bcc_fe), 3)


@pytest.mark.integration
def test_relax_returns_structure(calc: EqV2Calculator, bcc_fe) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result
    assert "trajectory" in result
