"""Tests for PhonopyAnalyzer."""

from __future__ import annotations

import pytest

pytest.importorskip("phonopy")

from materialsframework.analysis.phonopy import PhonopyAnalyzer
from materialsframework.transformations.phonopy import PhonopyDisplacementTransformation


@pytest.fixture(scope="module")
def analyzer(calc):
    """PhonopyAnalyzer with CHGNet calculator."""
    return PhonopyAnalyzer(calculator=calc)


@pytest.fixture(scope="module")
def result(analyzer, bcc_fe):
    """Single phonon calculation result shared by all result-checking tests."""
    return analyzer.calculate(bcc_fe, is_relaxed=True, mesh=10, pdos_mesh=5)


def test_default_params() -> None:
    """Analyzer initialises with the correct default state."""
    analyzer = PhonopyAnalyzer()
    assert analyzer._calculator is None
    assert analyzer._phonopy_transformation is None


def test_phonopy_transformation_lazy_property() -> None:
    """Accessing .phonopy_transformation creates a PhonopyDisplacementTransformation."""
    analyzer = PhonopyAnalyzer()
    assert isinstance(analyzer.phonopy_transformation, PhonopyDisplacementTransformation)


@pytest.mark.integration
def test_calculate_returns_keys(result) -> None:
    """calculate() returns a dict with total_dos, thermal_properties, and projected_dos."""
    for key in ("total_dos", "thermal_properties", "projected_dos"):
        assert key in result


@pytest.mark.integration
def test_total_dos_has_frequency_and_dos(result) -> None:
    """total_dos dict contains frequency_points and total_dos arrays."""
    assert "frequency_points" in result["total_dos"]
    assert "total_dos" in result["total_dos"]


@pytest.mark.integration
def test_thermal_properties_has_expected_keys(result) -> None:
    """thermal_properties dict contains temperatures and entropy."""
    assert "temperatures" in result["thermal_properties"]


@pytest.mark.integration
def test_projected_dos_has_expected_keys(result) -> None:
    """projected_dos dict contains frequency_points."""
    assert "frequency_points" in result["projected_dos"]
