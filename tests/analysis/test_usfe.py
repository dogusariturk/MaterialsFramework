"""Tests for USFEAnalyzer."""

from __future__ import annotations

import pytest

from materialsframework.analysis.usfe import USFEAnalyzer
from materialsframework.transformations.usfe import USFETransformation

_DEFAULT_NUM_STEPS = 11
_THREE_STEPS = 3


def test_default_params() -> None:
    """Analyzer stores expected default values."""
    analyzer = USFEAnalyzer()
    assert analyzer.slip_plane == "110"
    assert analyzer.start == pytest.approx(0.0)
    assert analyzer.stop == pytest.approx(1.0)
    assert analyzer.num_steps == _DEFAULT_NUM_STEPS
    assert analyzer._calculator is None
    assert analyzer._usfe_transformation is None


def test_invalid_slip_plane_raises() -> None:
    """Unsupported slip planes raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported slip_plane"):
        USFEAnalyzer(slip_plane="123")


def test_usfe_transformation_lazy_property() -> None:
    """Accessing .usfe_transformation creates a default USFETransformation."""
    analyzer = USFEAnalyzer(slip_plane="112", num_steps=_THREE_STEPS)
    assert isinstance(analyzer.usfe_transformation, USFETransformation)
    assert analyzer.usfe_transformation.slip_plane == "112"


@pytest.mark.integration
def test_calculate_returns_expected_output_keys(calc, bcc_fe) -> None:
    """calculate() returns all expected USFE result fields."""
    analyzer = USFEAnalyzer(calculator=calc, slip_plane="110", num_steps=_THREE_STEPS)
    result = analyzer.calculate(bcc_fe, is_relaxed=True)

    for key in (
        "gsfe_curve",
        "usfe_mJ_m2",
        "usfe_displacement_frac",
        "slip_plane",
        "num_steps",
    ):
        assert key in result


@pytest.mark.integration
def test_calculate_accepts_ase_atoms(calc, ase_bcc_fe) -> None:
    """calculate() accepts ase.Atoms as input."""
    analyzer = USFEAnalyzer(calculator=calc, num_steps=_THREE_STEPS)
    result = analyzer.calculate(ase_bcc_fe, is_relaxed=True)
    assert result["num_steps"] == _THREE_STEPS


@pytest.mark.integration
def test_calculate_extracts_usfe_from_gsfe_curve(calc, bcc_fe) -> None:
    """USFE equals the maximum gamma value in the reported GSFE curve."""
    analyzer = USFEAnalyzer(calculator=calc, num_steps=_THREE_STEPS)
    result = analyzer.calculate(bcc_fe, is_relaxed=True)

    assert result["usfe_mJ_m2"] == pytest.approx(max(point["gamma_mJ_m2"] for point in result["gsfe_curve"]))
