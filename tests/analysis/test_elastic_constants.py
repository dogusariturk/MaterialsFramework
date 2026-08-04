"""Tests for ElasticConstantsAnalyzer."""

from __future__ import annotations

import math

import pytest
from ase import Atoms
from ase.cell import Cell

from materialsframework.analysis.elastic_constants import ElasticConstantsAnalyzer
from materialsframework.tools.elastic import get_lattice_type
from materialsframework.transformations.elastic_constants import ElasticConstantsDeformationTransformation


def _make_atoms(a, b, c, alpha, beta, gamma) -> Atoms:
    atoms = Atoms("H", positions=[[0, 0, 0]])
    atoms.cell = Cell.fromcellpar([a, b, c, alpha, beta, gamma])
    atoms.pbc = True
    return atoms


@pytest.mark.parametrize(
    "atoms, expected_brav",
    [
        (_make_atoms(2.83, 2.83, 2.83, 90, 90, 90), "Cubic"),
        (_make_atoms(3.25, 3.25, 4.95, 90, 90, 90), "Tetragonal"),
        (_make_atoms(3.21, 3.21, 5.21, 90, 90, 120), "Hexagonal"),
        (_make_atoms(3.0, 4.0, 5.0, 90, 90, 90), "Orthorombic"),
        (_make_atoms(4.0, 4.0, 4.0, 70, 70, 70), "Trigonal"),
        (_make_atoms(3.0, 4.0, 5.0, 90, 110, 90), "Monoclinic"),
        (_make_atoms(3.0, 4.0, 5.0, 80, 90, 100), "Triclinic"),
    ],
)
def test_get_lattice_type(atoms, expected_brav) -> None:
    """get_lattice_type correctly classifies all seven crystal systems."""
    _, brav, _, _ = get_lattice_type(atoms)
    assert brav == expected_brav


@pytest.fixture(scope="module")
def analyzer(calc):
    """ElasticConstantsAnalyzer with reduced deformations for fast tests."""
    return ElasticConstantsAnalyzer(num_deform=3, max_deform=1.0, calculator=calc)


@pytest.fixture(scope="module")
def result(analyzer, bcc_fe):
    """Single elastic constant calculation shared by all result-checking tests."""
    return analyzer.calculate(bcc_fe, is_relaxed=True)


def test_default_params() -> None:
    """Analyzer stores the correct default parameters."""
    analyzer = ElasticConstantsAnalyzer()
    assert analyzer.num_deform == 5
    assert analyzer.max_deform == pytest.approx(2)
    assert analyzer._calculator is None
    assert analyzer._elastic_constant_transformation is None


def test_elastic_constants_transformation_lazy_property() -> None:
    """Accessing .elastic_constants_transformation creates an ElasticConstantsDeformationTransformation."""
    analyzer = ElasticConstantsAnalyzer()
    assert isinstance(analyzer.elastic_constants_transformation, ElasticConstantsDeformationTransformation)


def test_calculate_raises_without_required_properties(bcc_fe) -> None:
    """calculate() raises if the calculator lacks the required 'energy'/'stress' properties, before doing any real work."""
    from materialsframework.calculators.random import RandomCalculator

    class _NoEnergyOrStressCalculator(RandomCalculator):
        AVAILABLE_PROPERTIES = ["forces"]

    analyzer = ElasticConstantsAnalyzer(calculator=_NoEnergyOrStressCalculator())
    with pytest.raises(ValueError, match="'energy' and 'stress'"):
        analyzer.calculate(bcc_fe)


@pytest.mark.integration
def test_calculate_contains_cij_for_cubic(result) -> None:
    """For a cubic structure the result contains C_11, C_12, and C_44."""
    for key in ("C_11", "C_12", "C_44"):
        assert key in result


@pytest.mark.integration
def test_calculate_returns_moduli(result) -> None:
    """calculate() returns all expected mechanical properties."""
    for key in (
        "youngs_modulus",
        "voigt_bulk_modulus",
        "voigt_shear_modulus",
        "voigt_reuss_hill_bulk_modulus",
        "voigt_reuss_hill_shear_modulus",
        "poisson_ratio",
        "pugh_ratio",
        "chen_vickers_hardness",
    ):
        assert key in result


@pytest.mark.integration
def test_bulk_modulus_positive(result) -> None:
    """Voigt bulk modulus should be positive for BCC Fe."""
    assert result["voigt_bulk_modulus"] > 0


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
