"""Calculator package for MaterialsFramework.

Individual calculator classes are lazily imported on attribute access.
Use ``get_calculator(name)`` for name-based lookup without triggering imports.
"""

from __future__ import annotations

from materialsframework._registry import lazy_getattr
from materialsframework.calculators.registry import get_calculator, list_calculators

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

_CALCULATOR_MAP: dict[str, tuple[str, str]] = {
    "AlignnCalculator": ("materialsframework.calculators.alignn", "AlignnCalculator"),
    "AllegroCalculator": ("materialsframework.calculators.allegro", "AllegroCalculator"),
    "AlphaNetCalculator": ("materialsframework.calculators.alphanet", "AlphaNetCalculator"),
    "CHGNetCalculator": ("materialsframework.calculators.chgnet", "CHGNetCalculator"),
    "DeePMDCalculator": ("materialsframework.calculators.deepmd", "DeePMDCalculator"),
    "EqnormCalculator": ("materialsframework.calculators.eqnorm", "EqnormCalculator"),
    "EqV2Calculator": ("materialsframework.calculators.eqv2", "EqV2Calculator"),
    "eSENCalculator": ("materialsframework.calculators.esen", "eSENCalculator"),
    "GPTFFCalculator": ("materialsframework.calculators.gptff", "GPTFFCalculator"),
    "GraceCalculator": ("materialsframework.calculators.grace", "GraceCalculator"),
    "HIENetCalculator": ("materialsframework.calculators.hienet", "HIENetCalculator"),
    "M3GNetCalculator": ("materialsframework.calculators.m3gnet", "M3GNetCalculator"),
    "MACECalculator": ("materialsframework.calculators.mace", "MACECalculator"),
    "MatRISCalculator": ("materialsframework.calculators.matris", "MatRISCalculator"),
    "MatterSimCalculator": ("materialsframework.calculators.mattersim", "MatterSimCalculator"),
    "MEGNetCalculator": ("materialsframework.calculators.megnet", "MEGNetCalculator"),
    "NequIPCalculator": ("materialsframework.calculators.nequip", "NequIPCalculator"),
    "NequixCalculator": ("materialsframework.calculators.nequix", "NequixCalculator"),
    "NewtonNetCalculator": ("materialsframework.calculators.newtonnet", "NewtonNetCalculator"),
    "ORBCalculator": ("materialsframework.calculators.orb", "ORBCalculator"),
    "PetMadCalculator": ("materialsframework.calculators.petmad", "PetMadCalculator"),
    "PosEGNNCalculator": ("materialsframework.calculators.posegnn", "PosEGNNCalculator"),
    "RandomCalculator": ("materialsframework.calculators.random", "RandomCalculator"),
    "SevenNetCalculator": ("materialsframework.calculators.sevennet", "SevenNetCalculator"),
    "TACECalculator": ("materialsframework.calculators.tace", "TACECalculator"),
    "UMACalculator": ("materialsframework.calculators.uma", "UMACalculator"),
    "VASPCalculator": ("materialsframework.calculators.vasp", "VASPCalculator"),
}

__all__ = [*_CALCULATOR_MAP, "get_calculator", "list_calculators"]  # noqa: PLE0604


def __getattr__(name: str) -> type:
    """Lazily import and return a calculator class by name.

    Enables attribute-style access to calculators (e.g., ``calculators.M3GNetCalculator``)
    without eagerly importing all sub-modules at package load time.

    Args:
        name (str): The calculator class name to look up.

    Returns:
        type: The requested calculator class.

    Raises:
        AttributeError: If ``name`` is not found in the calculator registry.
    """
    return lazy_getattr(name, __name__, _CALCULATOR_MAP)
