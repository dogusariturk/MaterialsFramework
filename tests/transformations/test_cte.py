"""Tests for CTETransformation."""

from __future__ import annotations

import pytest

from materialsframework.transformations.cte import CTETransformation

_PAIR_COUNT = 2
_TEST_STEPS = 50


def test_default_params() -> None:
    """CTETransformation stores defaults."""
    transformation = CTETransformation()
    assert transformation.ensemble == "npt_berendsen"
    assert transformation.pressure == pytest.approx(1.0)


def test_apply_transformation_returns_tasks_and_structures(bcc_fe) -> None:
    """apply_transformation() returns one task and one structure per temperature."""
    transformation = CTETransformation()
    result = transformation.apply_transformation(bcc_fe, temperatures=[300.0, 400.0], steps=_TEST_STEPS)

    assert len(result["tasks"]) == _PAIR_COUNT
    assert len(result["structures"]) == _PAIR_COUNT
    assert result["tasks"][0]["temperature"] == pytest.approx(300.0)
    assert result["tasks"][1]["temperature"] == pytest.approx(400.0)
    assert result["tasks"][0]["steps"] == _TEST_STEPS


def test_apply_transformation_accepts_ase_atoms(ase_bcc_fe) -> None:
    """apply_transformation() accepts ase.Atoms input."""
    transformation = CTETransformation()
    result = transformation.apply_transformation(ase_bcc_fe, temperatures=[300.0, 350.0], steps=3)
    assert len(result["tasks"]) == _PAIR_COUNT


@pytest.mark.parametrize(
    "temperatures",
    [
        [],
        [300.0, -50.0],
        [300.0, 0.0],
        [300.0, float("nan")],
        "300,400",
    ],
)
def test_apply_transformation_validates_temperatures(bcc_fe, temperatures) -> None:  # noqa: ANN001
    """apply_transformation() raises ValueError for invalid temperature inputs."""
    transformation = CTETransformation()
    with pytest.raises(ValueError):
        transformation.apply_transformation(bcc_fe, temperatures=temperatures, steps=3)


def test_apply_transformation_requires_positive_steps(bcc_fe) -> None:
    """apply_transformation() rejects non-positive MD step counts."""
    transformation = CTETransformation()
    with pytest.raises(ValueError, match="steps must be a positive integer"):
        transformation.apply_transformation(bcc_fe, temperatures=[300.0, 400.0], steps=0)
