"""Tests for HSolubilityAnalyzer."""

from __future__ import annotations

import pytest

from materialsframework.analysis.h_solubility import HSolubilityAnalyzer
from materialsframework.transformations.h_solubility import HSolubilityTransformation


def test_default_params() -> None:
    """Analyzer initializes with lazy dependencies."""
    analyzer = HSolubilityAnalyzer()
    assert analyzer._calculator is None
    assert analyzer._h_solubility_transformation is None


def test_h_solubility_transformation_lazy_property() -> None:
    """Accessing lazy property constructs default transformation."""
    analyzer = HSolubilityAnalyzer()
    assert isinstance(analyzer.h_solubility_transformation, HSolubilityTransformation)


def test_calculate_raises_without_energy_property(bcc_fe) -> None:
    """calculate() raises if the calculator lacks the 'energy' property, before doing any real work."""
    from materialsframework.calculators.random import RandomCalculator

    class _NoEnergyCalculator(RandomCalculator):
        AVAILABLE_PROPERTIES = ["forces"]

    analyzer = HSolubilityAnalyzer(calculator=_NoEnergyCalculator())
    with pytest.raises(ValueError, match="'energy'"):
        analyzer.calculate(bcc_fe)


@pytest.mark.integration
def test_calculate_returns_solution_energy_fields(calc, bcc_fe) -> None:
    """Analyzer returns insertion energies and a solution energy field."""
    analyzer = HSolubilityAnalyzer(calculator=calc)
    result = analyzer.calculate(
        bcc_fe,
        site_types=("octahedral", "tetrahedral"),
        max_sites_per_type=1,
        is_relaxed=True,
    )

    assert "octahedral_insertion_energies" in result
    assert "tetrahedral_insertion_energies" in result
    assert "preferred_site_type" in result
    assert "solution_energy" in result


@pytest.mark.integration
def test_calculate_allows_single_site_family(calc, bcc_fe) -> None:
    """Analyzer supports evaluating only one requested site family."""
    analyzer = HSolubilityAnalyzer(calculator=calc)
    result = analyzer.calculate(
        bcc_fe,
        site_types=("tetrahedral",),
        max_sites_per_type=1,
        is_relaxed=True,
    )

    assert result["octahedral_insertion_energies"] == []
    assert len(result["tetrahedral_insertion_energies"]) == 1
