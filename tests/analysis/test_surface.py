"""Tests for SurfaceAnalyzer and SurfaceTransformation."""

from __future__ import annotations

import pytest

from materialsframework.analysis.surface import SurfaceAnalyzer
from materialsframework.calculators.random import RandomCalculator
from materialsframework.constants import EV_A2_TO_J_M2
from materialsframework.transformations.surface import SurfaceTransformation


@pytest.fixture(scope="module")
def analyzer(calc):
    """SurfaceAnalyzer with a small slab/vacuum size for fast integration tests."""
    return SurfaceAnalyzer(calculator=calc, miller_index=(1, 1, 0), min_slab_size=6.0, min_vacuum_size=8.0)


@pytest.fixture(scope="module")
def result(analyzer, bcc_fe):
    """Single surface energy calculation result shared by all result-checking tests."""
    return analyzer.calculate(bcc_fe, is_relaxed=True)


def test_default_params() -> None:
    """SurfaceAnalyzer stores the documented default values."""
    analyzer = SurfaceAnalyzer()
    assert analyzer.miller_index == (1, 1, 0)
    assert analyzer.min_slab_size == pytest.approx(10.0)
    assert analyzer.min_vacuum_size == pytest.approx(10.0)
    assert analyzer.center_slab is True
    assert analyzer.in_unit_planes is False
    assert analyzer.primitive is False
    assert analyzer.symmetrize is True
    assert analyzer._calculator is None
    assert analyzer._surface_transformation is None


def test_surface_transformation_lazy_property() -> None:
    """Accessing .surface_transformation creates a SurfaceTransformation seeded from the analyzer's params."""
    analyzer = SurfaceAnalyzer(miller_index=(1, 0, 0), min_slab_size=5.0, symmetrize=False)
    transformation = analyzer.surface_transformation
    assert isinstance(transformation, SurfaceTransformation)
    assert transformation.miller_index == (1, 0, 0)
    assert transformation.min_slab_size == pytest.approx(5.0)
    assert transformation.symmetrize is False


def test_calculate_raises_without_energy_property(bcc_fe) -> None:
    """calculate() raises if the calculator lacks the 'energy' property, before doing any real work."""

    class _NoEnergyCalculator(RandomCalculator):
        AVAILABLE_PROPERTIES = ["forces"]

    analyzer = SurfaceAnalyzer(calculator=_NoEnergyCalculator())
    with pytest.raises(ValueError, match="'energy'"):
        analyzer.calculate(bcc_fe)


def test_calculate_returns_expected_keys(bcc_fe) -> None:
    """calculate() returns bulk_structure/bulk_energy/bulk_energy_per_atom/slabs, each slab entry fully populated."""
    analyzer = SurfaceAnalyzer(calculator=RandomCalculator(), miller_index=(1, 1, 0))
    result = analyzer.calculate(bcc_fe, is_relaxed=True)

    for key in ("bulk_structure", "bulk_energy", "bulk_energy_per_atom", "slabs"):
        assert key in result

    assert len(result["slabs"]) > 0
    for slab_result in result["slabs"]:
        for key in ("slab", "relaxed_slab", "slab_energy", "slab_area", "gamma", "gamma_J_m2"):
            assert key in slab_result


def test_calculate_restores_relax_cell(bcc_fe) -> None:
    """calculate() restores the calculator's relax_cell setting afterward, instead of leaking it."""
    random_calc = RandomCalculator()
    random_calc.relax_cell = True
    analyzer = SurfaceAnalyzer(calculator=random_calc, miller_index=(1, 1, 0))

    analyzer.calculate(bcc_fe, is_relaxed=True)

    assert random_calc.relax_cell is True


def test_calculate_gamma_j_m2_matches_conversion(bcc_fe) -> None:
    """gamma_J_m2 is gamma (eV/Angstrom^2) converted with the shared eV/Angstrom^2 -> J/m^2 factor."""
    analyzer = SurfaceAnalyzer(calculator=RandomCalculator(), miller_index=(1, 1, 0))
    result = analyzer.calculate(bcc_fe, is_relaxed=True)

    for slab_result in result["slabs"]:
        assert slab_result["gamma_J_m2"] == pytest.approx(slab_result["gamma"] * EV_A2_TO_J_M2)


@pytest.mark.integration
def test_surface_energy_is_positive(result) -> None:
    """The BCC Fe (110) surface energy should be positive, since creating a surface costs energy."""
    for slab_result in result["slabs"]:
        assert slab_result["gamma"] > 0
        assert slab_result["gamma_J_m2"] > 0


@pytest.mark.integration
def test_bulk_energy_per_atom_matches_structure(result, bcc_fe) -> None:
    """bulk_energy_per_atom is bulk_energy divided by the number of sites in the relaxed bulk structure."""
    assert result["bulk_energy_per_atom"] == pytest.approx(result["bulk_energy"] / result["bulk_structure"].num_sites)
