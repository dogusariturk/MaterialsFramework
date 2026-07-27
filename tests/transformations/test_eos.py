"""Tests for EOSTransformation."""

from __future__ import annotations

import pytest
from pymatgen.core import Structure

from materialsframework.transformations.eos import EOSTransformation


def test_default_params() -> None:
    """EOSTransformation stores the correct default start, stop, and num."""
    t = EOSTransformation()
    assert t._strains[0] == pytest.approx(-0.1)
    assert t._strains[-1] == pytest.approx(0.1)
    assert len(t._strains) == 11


def test_custom_params() -> None:
    """Custom start/stop/num produces the correct number of strains."""
    t = EOSTransformation(start=-0.05, stop=0.05, num=5)
    assert len(t._strains) == 5


@pytest.mark.parametrize("num", [0, 1, 2])
def test_num_below_minimum_raises(num) -> None:
    """A `num` below 3 raises a clear ValueError instead of failing later during EOS fitting."""
    with pytest.raises(ValueError, match="num must be at least 3"):
        EOSTransformation(num=num)


def test_apply_transformation_populates_structures(bcc_fe) -> None:
    """apply_transformation() returns structures with num entries."""
    t = EOSTransformation(start=-0.02, stop=0.02, num=4)
    result = t.apply_transformation(bcc_fe)
    assert len(result) == 4


def test_apply_transformation_structures_are_pymatgen(bcc_fe) -> None:
    """Each entry in the returned structures is a pymatgen Structure."""
    t = EOSTransformation(start=-0.02, stop=0.02, num=3)
    result = t.apply_transformation(bcc_fe)
    for s in result:
        assert isinstance(s, Structure)


def test_apply_transformation_preserves_site_count(bcc_fe) -> None:
    """Deformed structures keep the same number of sites as the original."""
    t = EOSTransformation(start=-0.02, stop=0.02, num=3)
    result = t.apply_transformation(bcc_fe)
    for s in result:
        assert len(s) == len(bcc_fe)


def test_apply_transformation_independent_calls(bcc_fe) -> None:
    """Calling apply_transformation() twice returns independent lists, not an accumulated one."""
    t = EOSTransformation(start=-0.02, stop=0.02, num=3)
    first = t.apply_transformation(bcc_fe)
    second = t.apply_transformation(bcc_fe)
    assert len(first) == 3
    assert len(second) == 3
    assert first is not second
