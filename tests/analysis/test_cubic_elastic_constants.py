"""Tests for CubicElasticConstantsAnalyzer."""

from __future__ import annotations

import math

import pytest

from materialsframework.analysis.cubic_elastic_constants import (
    CubicElasticConstantsAnalyzer,
)
from materialsframework.transformations.cubic_elastic_constants import CubicElasticConstantsDeformationTransformation


@pytest.fixture(scope="module")
def analyzer(calc):
    """CubicElasticConstantsAnalyzer with a small deformation range."""
    return CubicElasticConstantsAnalyzer(
        delta_max=0.02,
        step_size=0.01,
        calculator=calc,
    )


@pytest.fixture(scope="module")
def result(analyzer, bcc_fe):
    """Single elastic constant calculation shared by all result-checking tests."""
    return analyzer.calculate(bcc_fe, is_relaxed=True)


def test_default_params() -> None:
    """Analyzer stores the correct default parameters."""
    analyzer = CubicElasticConstantsAnalyzer()
    assert analyzer.eos_name == "birch_murnaghan"
    assert analyzer.delta_max == pytest.approx(0.05)
    assert analyzer.step_size == pytest.approx(0.01)
    assert analyzer._calculator is None
    assert analyzer._cubic_transformation is None


def test_cubic_transformation_lazy_property() -> None:
    """Accessing .cubic_transformation creates a CubicElasticConstantsDeformationTransformation."""
    analyzer = CubicElasticConstantsAnalyzer()
    assert isinstance(analyzer.cubic_transformation, CubicElasticConstantsDeformationTransformation)


def test_calculate_raises_without_energy_property(bcc_fe) -> None:
    """calculate() raises if the calculator lacks the 'energy' property, before doing any real work."""
    from materialsframework.calculators.random import RandomCalculator

    class _NoEnergyCalculator(RandomCalculator):
        AVAILABLE_PROPERTIES = ["forces"]

    analyzer = CubicElasticConstantsAnalyzer(calculator=_NoEnergyCalculator())
    with pytest.raises(ValueError, match="'energy'"):
        analyzer.calculate(bcc_fe)


@pytest.mark.integration
def test_calculate_returns_elastic_constants(result) -> None:
    """calculate() returns C11, C12, and C44 as floats."""
    for key in ("C11", "C12", "C44"):
        assert key in result
        assert isinstance(result[key], float)


@pytest.mark.integration
def test_calculate_returns_mechanical_properties(result) -> None:
    """calculate() returns all expected derived moduli and ratios."""
    for key in (
        "youngs_modulus",
        "voigt_bulk_modulus",
        "voigt_shear_modulus",
        "reuss_bulk_modulus",
        "reuss_shear_modulus",
        "voigt_reuss_hill_bulk_modulus",
        "voigt_reuss_hill_shear_modulus",
        "poisson_ratio",
        "pugh_ratio",
        "chen_vickers_hardness",
    ):
        assert key in result


@pytest.mark.integration
def test_bulk_modulus_positive(result) -> None:
    """VRH bulk modulus should be positive for BCC Fe."""
    assert result["voigt_reuss_hill_bulk_modulus"] > 0


@pytest.mark.integration
def test_c11_greater_than_c12(result) -> None:
    """BCC Fe must satisfy the mechanical stability criterion C11 > C12."""
    assert result["C11"] > result["C12"]


@pytest.mark.integration
def test_chen_vickers_hardness_matches_formula(result) -> None:
    """chen_vickers_hardness follows Hv = 2 * (k^2 * G)^0.585 - 3 for the returned Pugh ratio and G_VRH.

    The Chen model is only defined for a positive k^2 * G; for a mechanically unstable fit (negative
    shear modulus) it legitimately evaluates to NaN, so that case is asserted explicitly rather than
    compared with pytest.approx (which treats NaN as never equal by default).
    """
    base = result["pugh_ratio"] ** 2 * result["voigt_reuss_hill_shear_modulus"]
    if base < 0:
        assert math.isnan(result["chen_vickers_hardness"])
    else:
        assert result["chen_vickers_hardness"] == pytest.approx(2.0 * base**0.585 - 3.0)
