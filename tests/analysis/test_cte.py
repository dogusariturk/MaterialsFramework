"""Tests for CTEAnalyzer."""

from __future__ import annotations

import pytest

from materialsframework.analysis.cte import CTEAnalyzer
from materialsframework.calculators.random import RandomCalculator
from materialsframework.transformations.cte import CTETransformation

_TWO_POINTS = 2


def test_default_params() -> None:
    """CTEAnalyzer stores temperatures/ensemble/pressure defaults."""
    analyzer = CTEAnalyzer()
    assert analyzer.temperatures == (300.0, 600.0, 900.0)
    assert analyzer.ensemble == "npt_berendsen"
    assert analyzer.pressure == pytest.approx(1.0)


def test_cte_transformation_lazy_property(bcc_fe) -> None:
    """Accessing cte_transformation creates a CTETransformation matching the configured temperatures."""
    analyzer = CTEAnalyzer(temperatures=[300.0, 400.0])
    transformation = analyzer.cte_transformation
    assert isinstance(transformation, CTETransformation)
    assert set(transformation.apply_transformation(bcc_fe)) == {300.0, 400.0}


def test_calculate_rejects_non_md_calculator(bcc_fe) -> None:
    """calculate() requires a BaseMDCalculator, not just any BaseCalculator."""
    analyzer = CTEAnalyzer(temperatures=[300.0, 400.0], calculator=RandomCalculator())
    with pytest.raises(ValueError, match="BaseMDCalculator"):
        analyzer.calculate(bcc_fe, steps=2)


def test_calculate_rejects_non_positive_steps(bcc_fe) -> None:
    """calculate() rejects non-positive MD step counts."""
    analyzer = CTEAnalyzer(temperatures=[300.0, 400.0])
    with pytest.raises(ValueError, match="steps must be a positive integer"):
        analyzer.calculate(bcc_fe, steps=0)


@pytest.mark.integration
def test_calculate_returns_structured_cte_output(calc, bcc_fe) -> None:
    """calculate() returns temperature/volume lists and CTE summary fields."""
    analyzer = CTEAnalyzer(temperatures=[300.0, 350.0], calculator=calc)
    result = analyzer.calculate(bcc_fe, steps=2)

    assert {"temperatures", "volumes", "cte", "cte_ppm"} <= result.keys()
    assert len(result["temperatures"]) == _TWO_POINTS
    assert len(result["volumes"]) == _TWO_POINTS


@pytest.mark.integration
def test_calculate_accepts_ase_atoms(calc, ase_bcc_fe) -> None:
    """calculate() accepts ase.Atoms input."""
    analyzer = CTEAnalyzer(temperatures=[300.0, 350.0], calculator=calc)
    result = analyzer.calculate(ase_bcc_fe, steps=2)
    assert len(result["volumes"]) == _TWO_POINTS


def test_calculate_rejects_non_distinct_temperatures(bcc_fe) -> None:
    """calculate() requires at least two distinct temperatures."""
    analyzer = CTEAnalyzer(temperatures=[300.0, 300.0])
    with pytest.raises(ValueError, match="two distinct temperatures"):
        analyzer.calculate(bcc_fe, steps=2)


def test_calculate_rejects_invalid_temperatures(bcc_fe) -> None:
    """calculate() propagates explicit invalid-temperature validation."""
    analyzer = CTEAnalyzer(temperatures=[300.0, -5.0])
    with pytest.raises(ValueError, match="greater than 0 K"):
        analyzer.calculate(bcc_fe, steps=2)
