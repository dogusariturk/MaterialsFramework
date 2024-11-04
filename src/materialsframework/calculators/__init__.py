"""
This module provides the calculator classes for the MaterialsFramework.

The module includes several calculators, each designed to interface with specific machine learning
potentials and perform material property calculations such as potential energy, forces, and stresses.
These calculators support advanced material simulations and structure relaxations.

This module serves as a centralized collection of calculator classes to streamline
the workflow of materials modeling and property prediction within the MaterialsFramework.
"""
from .chgnet import CHGNetCalculator
from .eqv2 import EqV2Calculator
from .m3gnet import M3GNetCalculator
from .mace import MACECalculator
from .megnet import MEGNetCalculator
from .orb import ORBCalculator
from .sevennet import SevenNetCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

__all__ = ["CHGNetCalculator",
           "EqV2Calculator",
           "M3GNetCalculator",
           "MACECalculator",
           "MEGNetCalculator",
           "ORBCalculator",
           "SevenNetCalculator"]
