"""Tests for the calculator registry (no ML extras required)."""

from __future__ import annotations

import pytest

from materialsframework.calculators.registry import get_calculator, list_calculators


def test_list_calculators_returns_sorted_list() -> None:
    """list_calculators returns a non-empty sorted list of strings."""
    names = list_calculators()
    assert isinstance(names, list)
    assert len(names) > 0
    assert names == sorted(names)


def test_list_calculators_contains_known() -> None:
    """Known calculators appear in the registry."""
    names = list_calculators()
    for expected in ("random", "mace", "chgnet", "vasp", "m3gnet", "petmad"):
        assert expected in names


def test_get_calculator_random() -> None:
    """get_calculator('random') returns a RandomCalculator without any ML dep."""
    from materialsframework.calculators.random import RandomCalculator

    calc = get_calculator("random")
    assert isinstance(calc, RandomCalculator)


def test_get_calculator_unknown_raises() -> None:
    """get_calculator raises ValueError for unknown names."""
    with pytest.raises(ValueError, match="Unknown calculator"):
        get_calculator("does_not_exist_xyz")


def test_get_calculator_passes_kwargs() -> None:
    """get_calculator forwards kwargs to the calculator's __init__."""
    from materialsframework.calculators.random import RandomCalculator

    fmax = 0.05
    calc = get_calculator("random", fmax=fmax)
    assert isinstance(calc, RandomCalculator)
    assert calc.fmax == fmax
