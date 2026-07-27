"""Tests for SurfaceTransformation."""

from __future__ import annotations

import pytest
from pymatgen.core.surface import Slab

from materialsframework.transformations.surface import SurfaceTransformation


def test_default_params() -> None:
    """SurfaceTransformation stores the documented default values."""
    t = SurfaceTransformation()
    assert t.miller_index == (1, 1, 0)
    assert t.min_slab_size == pytest.approx(10.0)
    assert t.min_vacuum_size == pytest.approx(10.0)
    assert t.center_slab is True
    assert t.in_unit_planes is False
    assert t.primitive is False
    assert t.symmetrize is True


def test_custom_params() -> None:
    """Custom constructor parameters are stored correctly."""
    t = SurfaceTransformation(
        miller_index=(1, 0, 0),
        min_slab_size=5.0,
        min_vacuum_size=15.0,
        center_slab=False,
        in_unit_planes=True,
        primitive=True,
        symmetrize=False,
    )
    assert t.miller_index == (1, 0, 0)
    assert t.min_slab_size == pytest.approx(5.0)
    assert t.min_vacuum_size == pytest.approx(15.0)
    assert t.center_slab is False
    assert t.in_unit_planes is True
    assert t.primitive is True
    assert t.symmetrize is False


def test_apply_transformation_returns_slabs(bcc_fe) -> None:
    """apply_transformation returns a non-empty list of pymatgen Slab objects."""
    t = SurfaceTransformation(miller_index=(1, 1, 0), min_slab_size=10.0, min_vacuum_size=10.0)
    slabs = t.apply_transformation(bcc_fe)

    assert len(slabs) > 0
    for slab in slabs:
        assert isinstance(slab, Slab)
        assert slab.miller_index == (1, 1, 0)
        assert slab.surface_area > 0


def test_apply_transformation_independent_calls(bcc_fe) -> None:
    """Calling apply_transformation() twice returns independent lists, not an accumulated one."""
    t = SurfaceTransformation(miller_index=(1, 1, 0))
    first = t.apply_transformation(bcc_fe)
    second = t.apply_transformation(bcc_fe)

    assert first is not second
    assert len(first) == len(second)


def test_apply_transformation_different_miller_index(bcc_fe) -> None:
    """Slabs generated for a different Miller index carry that index."""
    t = SurfaceTransformation(miller_index=(1, 0, 0))
    slabs = t.apply_transformation(bcc_fe)

    assert len(slabs) > 0
    for slab in slabs:
        assert slab.miller_index == (1, 0, 0)
