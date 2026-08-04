"""Tests for EOSAnalyzer and EOSTransformation."""

from __future__ import annotations

import pytest

from materialsframework.analysis.eos import EOSAnalyzer
from materialsframework.transformations.eos import EOSTransformation


@pytest.fixture(scope="module")
def analyzer(calc):
    """EOSAnalyzer with a small deformation range for fast integration tests."""
    return EOSAnalyzer(calculator=calc, start=-0.02, stop=0.02, num=5)


@pytest.fixture(scope="module")
def result(analyzer, bcc_fe):
    """Single EOS calculation result shared by all result-checking tests."""
    return analyzer.calculate(bcc_fe, is_relaxed=True)


def test_eos_transformation_generates_structures(bcc_fe) -> None:
    """EOSTransformation produces the expected number of deformed structures."""
    t = EOSTransformation(start=-0.02, stop=0.02, num=3)
    result = t.apply_transformation(bcc_fe)
    assert len(result) == 3


def test_eos_analyzer_default_params() -> None:
    """EOSAnalyzer stores default parameters correctly."""
    analyzer = EOSAnalyzer()
    assert analyzer.start == pytest.approx(-0.1)
    assert analyzer.stop == pytest.approx(0.1)
    assert analyzer.num == 11
    assert analyzer.eos_name == "birch_murnaghan"


def test_eos_transformation_lazy_property() -> None:
    """Accessing .eos_transformation creates an EOSTransformation instance."""
    analyzer = EOSAnalyzer(start=-0.02, stop=0.02, num=3)
    assert isinstance(analyzer.eos_transformation, EOSTransformation)


def test_calculate_raises_without_energy_property(bcc_fe) -> None:
    """calculate() raises if the calculator lacks the 'energy' property, before doing any real work."""
    from materialsframework.calculators.random import RandomCalculator

    class _NoEnergyCalculator(RandomCalculator):
        AVAILABLE_PROPERTIES = ["forces"]

    analyzer = EOSAnalyzer(calculator=_NoEnergyCalculator())
    with pytest.raises(ValueError, match="'energy'"):
        analyzer.calculate(bcc_fe)


@pytest.mark.integration
def test_eos_calculate_returns_keys(result) -> None:
    """EOSAnalyzer.calculate returns a dict with all expected keys."""
    for key in ("volumes", "energies", "e0", "b0", "b0_GPa", "b1", "v0"):
        assert key in result


@pytest.mark.integration
def test_eos_bulk_modulus_positive(result) -> None:
    """Bulk modulus should be positive for BCC Fe."""
    assert result["b0_GPa"] > 0
