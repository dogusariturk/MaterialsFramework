"""Tools package for MaterialsFramework.

Individual tool classes are lazily imported on attribute access.
Use ``get_tool(name)`` for name-based lookup without triggering imports.
"""

from __future__ import annotations

from materialsframework._registry import lazy_getattr
from materialsframework.tools.registry import get_tool, list_tools

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

_TOOLS_MAP: dict[str, tuple[str, str]] = {
    "BondLatticeParameter": ("materialsframework.tools.bond_lattice_parameter", "BondLatticeParameter"),
    "ClusterExpansion": ("materialsframework.tools.cluster_expansion", "ClusterExpansion"),
    "CoherentStabilityMap": ("materialsframework.tools.stability_map", "CoherentStabilityMap"),
    "MaterialParameters": ("materialsframework.tools.cahn_hilliard", "MaterialParameters"),
    "PhaseFieldModel": ("materialsframework.tools.cahn_hilliard", "PhaseFieldModel"),
    "SimulationGrid": ("materialsframework.tools.cahn_hilliard", "SimulationGrid"),
    "Sqs2tdb": ("materialsframework.tools.sqs2tdb", "Sqs2tdb"),
    "SqsGenerator": ("materialsframework.tools.sqsgen", "SqsGenerator"),
    "StabilityMap": ("materialsframework.tools.stability_map", "StabilityMap"),
    "TrajectoryObserver": ("materialsframework.tools.trajectory", "TrajectoryObserver"),
}

__all__ = [*_TOOLS_MAP, "get_tool", "list_tools"]  # noqa: PLE0604


def __getattr__(name: str) -> type:
    """Lazily import and return a tool class by name.

    Enables attribute-style access to tools (e.g., ``tools.ClusterExpansion``)
    without eagerly importing all sub-modules at package load time.

    Args:
        name (str): The tool class name to look up.

    Returns:
        type: The requested tool class.

    Raises:
        AttributeError: If ``name`` is not found in the tools map.
    """
    return lazy_getattr(name, __name__, _TOOLS_MAP)
