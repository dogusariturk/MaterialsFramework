"""Tests for BainPathAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest

from materialsframework.analysis.bain import BainPathAnalyzer
from materialsframework.transformations.bain import BainDisplacementTransformation


@pytest.fixture(scope="module")
def analyzer(calc):
    """BainPathAnalyzer with a small c/a range for fast integration tests."""
    return BainPathAnalyzer(start=0.9, stop=1.1, step=0.1, calculator=calc)


@pytest.fixture(scope="module")
def result(analyzer, bcc_fe):
    """Single Bain path calculation result shared by all result-checking tests."""
    return analyzer.calculate(bcc_fe, is_relaxed=True)


def test_default_params() -> None:
    """Analyzer stores the correct default start, stop, and step values."""
    analyzer = BainPathAnalyzer()
    assert analyzer.start == pytest.approx(0.89)
    assert analyzer.stop == pytest.approx(1.5)
    assert analyzer.step == pytest.approx(0.01)
    assert analyzer._calculator is None
    assert analyzer._bain_transformation is None


def test_bain_transformation_lazy_property() -> None:
    """Accessing .bain_transformation creates a BainDisplacementTransformation."""
    analyzer = BainPathAnalyzer(start=0.9, stop=1.1, step=0.1)
    assert isinstance(analyzer.bain_transformation, BainDisplacementTransformation)


def test_calculate_returns_keys_with_random_calc(bcc_fe) -> None:
    """calculate() returns required keys without any ML dependency."""
    from materialsframework.calculators.random import RandomCalculator

    analyzer = BainPathAnalyzer(start=0.9, stop=1.1, step=0.1, calculator=RandomCalculator())
    result = analyzer.calculate(bcc_fe, is_relaxed=True)
    assert "c_a_list" in result
    assert "energy_list" in result


def test_calculate_accepts_ase_atoms(ase_bcc_fe) -> None:
    """calculate() accepts ase.Atoms in addition to pymatgen Structure."""
    from materialsframework.calculators.random import RandomCalculator

    analyzer = BainPathAnalyzer(start=0.9, stop=1.1, step=0.1, calculator=RandomCalculator())
    result = analyzer.calculate(ase_bcc_fe, is_relaxed=True)
    assert "c_a_list" in result


@pytest.mark.integration
def test_calculate_returns_keys(result) -> None:
    """calculate() returns a dict with 'c_a_list' and 'energy_list' keys."""
    assert "c_a_list" in result
    assert "energy_list" in result


@pytest.mark.integration
def test_calculate_list_lengths_match(result) -> None:
    """c_a_list and energy_list have equal length and contain at least one point."""
    assert len(result["c_a_list"]) == len(result["energy_list"])
    assert len(result["c_a_list"]) > 0


@pytest.mark.integration
def test_energy_list_contains_floats(result) -> None:
    """Every entry in energy_list is a float."""
    for e in result["energy_list"]:
        assert isinstance(e, (float, np.floating))
