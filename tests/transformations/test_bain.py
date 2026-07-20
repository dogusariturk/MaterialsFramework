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
