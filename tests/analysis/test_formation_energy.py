"""Tests for FormationEnergyAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest

from materialsframework.analysis.formation_energy import FormationEnergyAnalyzer
from materialsframework.transformations.formation_energy import FormationEnergyTransformation


@pytest.fixture(scope="module")
def analyzer(calc):
    """FormationEnergyAnalyzer with CHGNet calculator."""
    return FormationEnergyAnalyzer(calculator=calc)


@pytest.fixture(scope="module")
def result(analyzer, l10_feni):
    """Single formation energy calculation shared by all result-checking tests."""
    return analyzer.calculate(l10_feni, is_relaxed=True)


def test_default_params() -> None:
    """Analyzer initialises with no calculator and no transformation."""
    analyzer = FormationEnergyAnalyzer()
    assert analyzer._calculator is None
    assert analyzer._formation_energy_transformation is None


def test_formation_energy_transformation_lazy_property() -> None:
    """Accessing .formation_energy_transformation creates a FormationEnergyTransformation."""
    analyzer = FormationEnergyAnalyzer()
    assert isinstance(analyzer.formation_energy_transformation, FormationEnergyTransformation)


def test_calculate_returns_key_with_random_calc(l10_feni) -> None:
    """calculate() returns formation_energy key without any ML dependency."""
    from materialsframework.calculators.random import RandomCalculator

    analyzer = FormationEnergyAnalyzer(calculator=RandomCalculator())
    result = analyzer.calculate(l10_feni, is_relaxed=True)
    assert "formation_energy" in result
    assert isinstance(result["formation_energy"], (float, np.floating))


def test_calculate_accepts_ase_atoms(ase_l10_feni) -> None:
    """calculate() accepts ase.Atoms in addition to pymatgen Structure."""
    from materialsframework.calculators.random import RandomCalculator

    analyzer = FormationEnergyAnalyzer(calculator=RandomCalculator())
    result = analyzer.calculate(ase_l10_feni, is_relaxed=True)
    assert "formation_energy" in result


@pytest.mark.integration
def test_single_element_raises(analyzer, bcc_fe) -> None:
    """calculate() raises ValueError when the structure has only one element type."""
    with pytest.raises(ValueError, match="at least two different elements"):
        analyzer.calculate(bcc_fe, is_relaxed=True)


@pytest.mark.integration
def test_calculate_returns_formation_energy(result) -> None:
    """calculate() returns a dict with a float 'formation_energy' key."""
    assert "formation_energy" in result
    assert isinstance(result["formation_energy"], (float, np.floating))


@pytest.mark.integration
def test_formation_energy_is_negative_for_feni(result) -> None:
    """FeNi L1_0 should have a negative formation energy (stable intermetallic)."""
    assert result["formation_energy"] < 0
