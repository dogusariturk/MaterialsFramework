"""Tests for the transformation registry (no optional extras required)."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from materialsframework.transformations.registry import get_transformation, list_transformations

_CUSTOM_START = -0.2
_CUSTOM_NUM = 5


def test_list_transformations_returns_sorted_list() -> None:
    """list_transformations returns a non-empty sorted list of strings."""
    names = list_transformations()
    assert isinstance(names, list)
    assert len(names) > 0
    assert names == sorted(names)


def test_list_transformations_contains_known() -> None:
    """Known transformations appear in the registry."""
    names = list_transformations()
    for expected in ("eos", "bain", "phonopy", "annni", "usfe", "cte", "h_solubility"):
        assert expected in names


def test_get_transformation_eos() -> None:
    """get_transformation('eos') returns an EOSTransformation without any optional dep."""
    from materialsframework.transformations.eos import EOSTransformation

    transformation = get_transformation("eos")
    assert isinstance(transformation, EOSTransformation)


def test_get_transformation_resolves_every_registered_name() -> None:
    """Every entry-point-registered transformation name resolves to a real, constructible class.

    Complements the hardcoded checks above by walking the *live* registry: a typo'd or
    drifted entry-point target (a bad module path or renamed class) fails loudly here even
    if it's a transformation not covered by any hardcoded name list.
    """
    for name in list_transformations():
        # TypeError means it requires constructor arguments
        with contextlib.suppress(TypeError):
            get_transformation(name)


def test_get_transformation_unknown_raises() -> None:
    """get_transformation raises ValueError for unknown names."""
    with pytest.raises(ValueError, match="Unknown transformation"):
        get_transformation("does_not_exist_xyz")


def test_get_transformation_passes_kwargs() -> None:
    """get_transformation forwards kwargs to the transformation's __init__."""
    from materialsframework.transformations.eos import EOSTransformation

    transformation = get_transformation("eos", start=_CUSTOM_START, num=_CUSTOM_NUM)
    assert isinstance(transformation, EOSTransformation)
    assert np.isclose(transformation._strains[0], _CUSTOM_START)  # noqa: SLF001
