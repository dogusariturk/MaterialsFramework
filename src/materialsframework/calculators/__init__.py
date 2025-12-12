"""This package provides the calculator classes for the MaterialsFramework.

The package includes several calculators, each designed to interface with specific machine learning
potentials and perform material property calculations such as potential energy, forces, and stresses.
These calculators support advanced material simulations and structure relaxations.

This package serves as a centralized collection of calculator classes to streamline
the workflow of materials modeling and property prediction within the MaterialsFramework.
"""
from .alignn import AlignnCalculator
from .alphanet import AlphaNetCalculator
from .chgnet import CHGNetCalculator
from .deepmd import DeePMDCalculator
from .eqnorm import EqnormCalculator
from .eqv2 import EqV2Calculator
from .esen import eSENCalculator
from .gptff import GPTFFCalculator
from .grace import GraceCalculator
from .hienet import HIENetCalculator
from .m3gnet import M3GNetCalculator
from .mace import MACECalculator
from .mattersim import MatterSimCalculator
from .megnet import MEGNetCalculator
from .nequip import NequIPCalculator
from .newtonnet import NewtonNetCalculator
from .orb import ORBCalculator
from .petmad import PetMadCalculator
from .posegnn import PosEGNNCalculator
from .sevennet import SevenNetCalculator
from .uma import UMACalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

__all__ = [
    "AlignnCalculator",
    "AlphaNetCalculator",
    "CHGNetCalculator",
    "DeePMDCalculator",
    "EqnormCalculator",
    "EqV2Calculator",
    "eSENCalculator",
    "GPTFFCalculator",
    "GraceCalculator",
    "HIENetCalculator",
    "M3GNetCalculator",
    "MACECalculator",
    "MatterSimCalculator",
    "MEGNetCalculator",
    "NequIPCalculator",
    "NewtonNetCalculator",
    "ORBCalculator",
    "PetMadCalculator",
    "PosEGNNCalculator",
    "SevenNetCalculator",
    "UMACalculator"
]
