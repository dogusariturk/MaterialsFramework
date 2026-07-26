"""Tests for BainDisplacementTransformation."""

from __future__ import annotations

import pytest
from pymatgen.core import Structure

from materialsframework.transformations.bain import BainDisplacementTransformation


def test_default_params() -> None:
    """Default c/a range and step are stored correctly."""
    t = BainDisplacementTransformation()
    assert t.c_a_ratios[0] == pytest.approx(0.89)
    assert len(t.c_a_ratios) > 0


@pytest.mark.parametrize(("start", "stop", "step"), [(0.0, 1.5, 0.01), (-0.1, 1.5, 0.01), (0.89, 0.0, 0.01)])
def test_non_positive_start_or_stop_raises(start, stop, step) -> None:
    """A non-positive `start` or `stop` raises, since c/a must stay a positive ratio."""
    with pytest.raises(ValueError, match="positive c/a ratios"):
        BainDisplacementTransformation(start=start, stop=stop, step=step)


@pytest.mark.parametrize("step", [0.0, -0.01])
def test_non_positive_step_raises(step) -> None:
    """A non-positive `step` raises rather than silently producing an empty/infinite range."""
    with pytest.raises(ValueError, match="`step` must be positive"):
        BainDisplacementTransformation(start=0.89, stop=1.5, step=step)


def test_custom_params() -> None:
    """Custom start/stop/step produce the expected number of c/a values."""
    t = BainDisplacementTransformation(start=0.9, stop=1.05, step=0.1)
    assert len(t.c_a_ratios) == 2  # [0.9, 1.0]


def test_apply_transformation_returns_structures(bcc_fe) -> None:
    """apply_transformation returns a dict of pymatgen Structures."""
    t = BainDisplacementTransformation(start=0.9, stop=1.05, step=0.1)
    result = t.apply_transformation(bcc_fe)
    assert len(result) == 2
    for struct in result.values():
        assert isinstance(struct, Structure)


def test_displaced_structure_has_same_sites(bcc_fe) -> None:
    """Deformed structures keep the same number of sites as the input."""
    t = BainDisplacementTransformation(start=0.9, stop=1.05, step=0.1)
    result = t.apply_transformation(bcc_fe)
    for struct in result.values():
        assert len(struct) == len(bcc_fe)


def test_apply_transformation_standardizes_primitive_cell(primitive_bcc_fe, bcc_fe) -> None:
    """A non-axis-aligned primitive cell is standardized to the conventional cell before deforming.

    Without standardization, the Cartesian-axis-aligned Bain strain would be applied directly to the
    primitive cell's non-orthogonal lattice vectors, producing a nonphysical deformation.
    """
    t = BainDisplacementTransformation(start=0.9, stop=1.05, step=0.1)
    result = t.apply_transformation(primitive_bcc_fe)
    for struct in result.values():
        assert len(struct) == len(bcc_fe)
        assert struct.lattice.is_orthogonal
