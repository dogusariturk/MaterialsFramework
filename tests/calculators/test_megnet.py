"""Integration tests for MEGNetCalculator.

MEGNetCalculator predicts formation energy only. It does not support
relax() or calculate() in the same way as force-field calculators.
"""

from __future__ import annotations

import pytest

pytest.importorskip("matgl")

from materialsframework.calculators.megnet import MEGNetCalculator


@pytest.fixture(scope="module")
def calc() -> MEGNetCalculator:
    """MEGNetCalculator with default model."""
    return MEGNetCalculator()


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES is exactly ['formation_energy']."""
    assert MEGNetCalculator.AVAILABLE_PROPERTIES == ["formation_energy"]


@pytest.mark.integration
def test_calculate_formation_energy(calc: MEGNetCalculator, bcc_fe) -> None:
    """calculate() returns a float formation energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "formation_energy" in result
    assert isinstance(result["formation_energy"], float)
