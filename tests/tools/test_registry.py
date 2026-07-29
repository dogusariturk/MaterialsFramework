"""Tests for the tool registry (no optional extras required)."""

from __future__ import annotations

import pytest

from materialsframework.tools.registry import get_tool, list_tools


def test_list_tools_returns_sorted_list() -> None:
    """list_tools returns a non-empty sorted list of strings."""
    names = list_tools()
    assert isinstance(names, list)
    assert len(names) > 0
    assert names == sorted(names)


def test_list_tools_contains_known() -> None:
    """Known tools appear in the registry."""
    names = list_tools()
    for expected in ("cluster_expansion", "stability_map", "sqs2tdb", "bond_lattice_parameter", "trajectory_observer"):
        assert expected in names


def test_get_tool_cluster_expansion() -> None:
    """get_tool('cluster_expansion') returns a ClusterExpansion instance."""
    from materialsframework.tools.cluster_expansion import ClusterExpansion

    tool = get_tool("cluster_expansion")
    assert isinstance(tool, ClusterExpansion)


def test_get_tool_unknown_raises() -> None:
    """get_tool raises ValueError for unknown names."""
    with pytest.raises(ValueError, match="Unknown tool"):
        get_tool("does_not_exist_xyz")


def test_get_tool_passes_kwargs() -> None:
    """get_tool forwards kwargs to the tool's __init__."""
    from materialsframework.tools.cluster_expansion import ClusterExpansion

    tool = get_tool("cluster_expansion", symprec=1e-3)
    assert isinstance(tool, ClusterExpansion)
    assert tool.symprec == 1e-3


def test_get_tool_trajectory_observer() -> None:
    """get_tool('trajectory_observer') returns a TrajectoryObserver instance."""
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from materialsframework.tools.trajectory import TrajectoryObserver

    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.calc = EMT()
    tool = get_tool("trajectory_observer", atoms=atoms)
    assert isinstance(tool, TrajectoryObserver)
