"""Tests for ClusterExpansion (no icet dependency required for unit tests)."""

from __future__ import annotations

import pytest

from materialsframework.tools.cluster_expansion import ClusterExpansion


def test_default_params() -> None:
    """ClusterExpansion stores correct default hyperparameters."""
    ce = ClusterExpansion()
    assert ce.symprec == pytest.approx(1e-5)
    assert ce.position_tolerance is None
    assert ce.is_relaxed is True
    assert ce.fit_method == "ardr"
    assert ce.standardize is True
    assert ce.validation_method == "k-fold"
    assert ce.n_splits == 10
    assert ce.check_condition is True
    assert ce.seed == 42
    assert ce.verbose is False


def test_initial_state() -> None:
    """ClusterExpansion starts with empty structures list and no fitted model."""
    ce = ClusterExpansion()
    assert ce._calculator is None
    assert ce.cluster_space is None
    assert ce.structure_container is None
    assert ce.cluster_expansion is None
    assert ce.structures == []


def test_calculator_lazy_property() -> None:
    """_calculator is None until .calculator is accessed."""
    ce = ClusterExpansion()
    assert ce._calculator is None


def test_custom_params() -> None:
    """Custom hyperparameters are stored correctly."""
    ce = ClusterExpansion(
        fit_method="ridge",
        validation_method="shuffle-split",
        n_splits=5,
        seed=0,
        verbose=True,
    )
    assert ce.fit_method == "ridge"
    assert ce.validation_method == "shuffle-split"
    assert ce.n_splits == 5
    assert ce.seed == 0
    assert ce.verbose is True
