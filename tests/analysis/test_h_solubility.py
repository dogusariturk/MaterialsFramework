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
