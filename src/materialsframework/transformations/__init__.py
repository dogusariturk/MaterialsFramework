"""Transformations package for MaterialsFramework.

Individual transformation classes are lazily imported on attribute access.
Use ``get_transformation(name)`` for name-based lookup without triggering imports.
"""

from __future__ import annotations

from materialsframework._registry import lazy_getattr
from materialsframework.transformations.registry import get_transformation, list_transformations

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

_TRANSFORMATION_MAP: dict[str, tuple[str, str]] = {
    "ANNNIStackingFaultTransformation": (
        "materialsframework.transformations.annni",
        "ANNNIStackingFaultTransformation",
    ),
    "BainDisplacementTransformation": ("materialsframework.transformations.bain", "BainDisplacementTransformation"),
    "CTETransformation": ("materialsframework.transformations.cte", "CTETransformation"),
    "CubicElasticConstantsDeformationTransformation": (
        "materialsframework.transformations.cubic_elastic_constants",
        "CubicElasticConstantsDeformationTransformation",
    ),
    "ElasticConstantsDeformationTransformation": (
        "materialsframework.transformations.elastic_constants",
        "ElasticConstantsDeformationTransformation",
    ),
    "EOSTransformation": ("materialsframework.transformations.eos", "EOSTransformation"),
    "FormationEnergyTransformation": (
        "materialsframework.transformations.formation_energy",
        "FormationEnergyTransformation",
    ),
    "HSolubilityTransformation": ("materialsframework.transformations.h_solubility", "HSolubilityTransformation"),
    "NEBTransformation": ("materialsframework.transformations.neb", "NEBTransformation"),
    "Phono3pyDisplacementTransformation": (
        "materialsframework.transformations.phono3py",
        "Phono3pyDisplacementTransformation",
    ),
    "PhonopyDisplacementTransformation": (
        "materialsframework.transformations.phonopy",
        "PhonopyDisplacementTransformation",
    ),
    "SBETransformation": ("materialsframework.transformations.sbe", "SBETransformation"),
    "SurfaceTransformation": ("materialsframework.transformations.surface", "SurfaceTransformation"),
    "USFETransformation": ("materialsframework.transformations.usfe", "USFETransformation"),
}

__all__ = [*_TRANSFORMATION_MAP, "get_transformation", "list_transformations"]  # noqa: PLE0604


def __getattr__(name: str) -> type:
    """Lazily import and return a transformation class by name.

    Enables attribute-style access to transformations (e.g., ``transformations.EOSTransformation``)
    without eagerly importing all sub-modules at package load time.

    Args:
        name (str): The transformation class name to look up.

    Returns:
        type: The requested transformation class.

    Raises:
        AttributeError: If ``name`` is not found in the transformation map.
    """
    return lazy_getattr(name, __name__, _TRANSFORMATION_MAP)
