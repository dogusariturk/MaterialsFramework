"""Analysis package for MaterialsFramework.

Individual analyzer classes are lazily imported on attribute access.
Use ``get_analyzer(name)`` for name-based lookup without triggering imports.
"""

from __future__ import annotations

from materialsframework._registry import lazy_getattr
from materialsframework.analysis.registry import get_analyzer, list_analyzers

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

_ANALYZER_MAP: dict[str, tuple[str, str]] = {
    "ANNNIStackingFaultAnalyzer": ("materialsframework.analysis.annni", "ANNNIStackingFaultAnalyzer"),
    "BainPathAnalyzer": ("materialsframework.analysis.bain", "BainPathAnalyzer"),
    "CTEAnalyzer": ("materialsframework.analysis.cte", "CTEAnalyzer"),
    "CubicElasticConstantsAnalyzer": ("materialsframework.analysis.cubic_elastic_constants", "CubicElasticConstantsAnalyzer"),
    "ElasticConstantsAnalyzer": ("materialsframework.analysis.elastic_constants", "ElasticConstantsAnalyzer"),
    "EOSAnalyzer": ("materialsframework.analysis.eos", "EOSAnalyzer"),
    "FormationEnergyAnalyzer": ("materialsframework.analysis.formation_energy", "FormationEnergyAnalyzer"),
    "HSolubilityAnalyzer": ("materialsframework.analysis.h_solubility", "HSolubilityAnalyzer"),
    "NEBAnalyzer": ("materialsframework.analysis.neb", "NEBAnalyzer"),
    "Phono3pyAnalyzer": ("materialsframework.analysis.phono3py", "Phono3pyAnalyzer"),
    "PhonopyAnalyzer": ("materialsframework.analysis.phonopy", "PhonopyAnalyzer"),
    "SBEAnalyzer": ("materialsframework.analysis.sbe", "SBEAnalyzer"),
    "SurfaceAnalyzer": ("materialsframework.analysis.surface", "SurfaceAnalyzer"),
    "USFEAnalyzer": ("materialsframework.analysis.usfe", "USFEAnalyzer"),
}

__all__ = [*_ANALYZER_MAP, "get_analyzer", "list_analyzers"]  # noqa: PLE0604


def __getattr__(name: str) -> type:
    """Lazily import and return an analyzer class by name.

    Enables attribute-style access to analyzers (e.g., ``analysis.EOSAnalyzer``)
    without eagerly importing all sub-modules at package load time.

    Args:
        name (str): The analyzer class name to look up.

    Returns:
        type: The requested analyzer class.

    Raises:
        AttributeError: If ``name`` is not found in the analyzer map.
    """
    return lazy_getattr(name, __name__, _ANALYZER_MAP)
