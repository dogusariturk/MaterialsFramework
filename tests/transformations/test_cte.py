"""Tests for CTETransformation."""

from __future__ import annotations

import pytest

from materialsframework.transformations.cte import CTETransformation

_PAIR_COUNT = 2


def test_apply_transformation_returns_one_structure_per_temperature(bcc_fe) -> None:
    """apply_transformation() returns one structure copy per temperature, keyed by temperature."""
    transformation = CTETransformation(temperatures=[300.0, 400.0])
    result = transformation.apply_transformation(bcc_fe)

    assert len(result) == _PAIR_COUNT
    assert set(result) == {300.0, 400.0}
    assert result[300.0] is not result[400.0]


def test_apply_transformation_accepts_ase_atoms(ase_bcc_fe) -> None:
    """apply_transformation() accepts ase.Atoms input."""
    transformation = CTETransformation(temperatures=[300.0, 350.0])
    result = transformation.apply_transformation(ase_bcc_fe)
    assert len(result) == _PAIR_COUNT


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
def test_init_validates_temperatures(temperatures) -> None:  # noqa: ANN001
    """CTETransformation raises ValueError for invalid temperature inputs at construction time."""
    with pytest.raises(ValueError):
        CTETransformation(temperatures=temperatures)
