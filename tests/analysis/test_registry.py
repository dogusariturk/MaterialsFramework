"""Tests for the analyzer registry (no optional extras required)."""

from __future__ import annotations

import pytest

from materialsframework.analysis.registry import get_analyzer, list_analyzers

_CUSTOM_START = -0.2


def test_list_analyzers_returns_sorted_list() -> None:
    """list_analyzers returns a non-empty sorted list of strings."""
    names = list_analyzers()
    assert isinstance(names, list)
    assert len(names) > 0
    assert names == sorted(names)


def test_list_analyzers_contains_known() -> None:
    """Known analyzers appear in the registry."""
    names = list_analyzers()
    for expected in ("eos", "bain", "phonopy", "annni", "usfe", "cte", "h_solubility"):
        assert expected in names


def test_get_analyzer_eos() -> None:
    """get_analyzer('eos') returns an EOSAnalyzer without any optional dep."""
    from materialsframework.analysis.eos import EOSAnalyzer

    analyzer = get_analyzer("eos")
    assert isinstance(analyzer, EOSAnalyzer)


def test_get_analyzer_unknown_raises() -> None:
    """get_analyzer raises ValueError for unknown names."""
    with pytest.raises(ValueError, match="Unknown analyzer"):
        get_analyzer("does_not_exist_xyz")


def test_get_analyzer_passes_kwargs() -> None:
    """get_analyzer forwards kwargs to the analyzer's __init__."""
    from materialsframework.analysis.eos import EOSAnalyzer

    analyzer = get_analyzer("eos", start=_CUSTOM_START)
    assert isinstance(analyzer, EOSAnalyzer)
    assert analyzer.start == _CUSTOM_START
