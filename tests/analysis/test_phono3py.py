"""Tests for Phono3pyAnalyzer."""

from __future__ import annotations

import pytest

pytest.importorskip("phono3py")

from materialsframework.analysis.phono3py import Phono3pyAnalyzer
from materialsframework.transformations.phono3py import Phono3pyDisplacementTransformation


@pytest.fixture(scope="module")
def analyzer(calc):
    """Phono3pyAnalyzer with CHGNet calculator."""
    return Phono3pyAnalyzer(calculator=calc)


@pytest.fixture(scope="module")
def result(analyzer, bcc_fe):
    """Single phono3py calculation result shared by all result-checking tests."""
    return analyzer.calculate(
        bcc_fe,
        is_relaxed=True,
        mesh=[5, 5, 5],
        t_min=300,
        t_max=300,
        t_step=1,
    )


def test_default_params() -> None:
    """Analyzer initialises with the correct default state."""
    analyzer = Phono3pyAnalyzer()
    assert analyzer._calculator is None
    assert analyzer._phono3py_transformation is None


def test_phono3py_transformation_lazy_property() -> None:
    """Accessing .phono3py_transformation creates a Phono3pyDisplacementTransformation."""
    analyzer = Phono3pyAnalyzer()
    assert isinstance(analyzer.phono3py_transformation, Phono3pyDisplacementTransformation)


@pytest.mark.integration
@pytest.mark.slow
def test_calculate_returns_thermal_conductivity(result) -> None:
    """calculate() returns a dict with a non-None 'thermal_conductivity' key."""
    assert "thermal_conductivity" in result
    assert result["thermal_conductivity"] is not None


@pytest.mark.integration
@pytest.mark.slow
def test_calculate_thermal_conductivity_is_not_none(result) -> None:
    """calculate() returns a non-None 'thermal_conductivity' value."""
    assert result["thermal_conductivity"] is not None
