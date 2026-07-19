"""Tests for VASPCalculator.

Functional tests require a VASP binary and a valid PP_PATH. Only the lazy-load
contract and initialization tests are run without VASP installed.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from materialsframework.calculators.vasp import VASPCalculator

_VASP_AVAILABLE = (
    any(os.getenv(var, "").strip() for var in ("ASE_VASP_COMMAND", "VASP_COMMAND", "VASP_SCRIPT"))
    or shutil.which("vasp_std") is not None
    or shutil.which("vasp") is not None
)


def test_lazy_load_contract() -> None:
    """_calculator must be None until the calculator property is accessed."""
    c = VASPCalculator()
    assert c._calculator is None


def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES includes 'energy' and 'forces'."""
    assert "energy" in VASPCalculator.AVAILABLE_PROPERTIES
    assert "forces" in VASPCalculator.AVAILABLE_PROPERTIES


def test_default_init_stores_none_params() -> None:
    """All optional VASP params default to None."""
    c = VASPCalculator()
    assert c._vasp_core.get("encut") is None
    assert c._vasp_core.get("xc") is None


@pytest.mark.integration
@pytest.mark.skipif(
    not _VASP_AVAILABLE,
    reason="VASP not configured (set ASE_VASP_COMMAND, VASP_COMMAND, or VASP_SCRIPT; or install vasp/vasp_std).",
)
def test_calculate_energy(bcc_fe) -> None:
    """calculate() returns a float energy for BCC Fe when VASP is available."""
    calc = VASPCalculator(xc="PBE", encut=400, kpts=(4, 4, 4))
    result = calc.calculate(bcc_fe)
    assert "energy" in result
    assert isinstance(result["energy"], (float, np.floating))
