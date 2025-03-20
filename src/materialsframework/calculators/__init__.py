"""
This module provides the calculator classes for the MaterialsFramework.

The module includes several calculators, each designed to interface with specific machine learning
potentials and perform material property calculations such as potential energy, forces, and stresses.
These calculators support advanced material simulations and structure relaxations.

This module serves as a centralized collection of calculator classes to streamline
the workflow of materials modeling and property prediction within the MaterialsFramework.
"""
from .alphanet import AlphaNetCalculator
from .chgnet import CHGNetCalculator
from .deepmd import DeePMDCalculator
from .divenet import DiveNetCalculator
from .eqv2 import EqV2Calculator
from .grace import GraceCalculator
from .m3gnet import M3GNetCalculator
from .mace import MACECalculator
from .mattersim import MatterSimCalculator
from .megnet import MEGNetCalculator
from .orb import ORBCalculator
from .sevennet import SevenNetCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

__all__ = ["AlphaNetCalculator",
           "CHGNetCalculator",
           "DeePMDCalculator",
           "DiveNetCalculator",
           "EqV2Calculator",
           "M3GNetCalculator",
           "GraceCalculator",
           "MACECalculator",
           "MatterSimCalculator",
           "MEGNetCalculator",
           "ORBCalculator",
           "SevenNetCalculator"]
