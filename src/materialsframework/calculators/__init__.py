"""
This package provides the calculator classes for the MaterialsFramework.

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
from .divenet import DiveNetCalculator
from .eqv2 import EqV2Calculator
from .esen import eSENCalculator
from .gptff import GPTFFCalculator
from .grace import GraceCalculator
from .m3gnet import M3GNetCalculator
from .mace import MACECalculator
from .mattersim import MatterSimCalculator
from .megnet import MEGNetCalculator
from .newtonnet import NewtonNetCalculator
from .orb import ORBCalculator
from .posegnn import PosEGNNCalculator
from .sevennet import SevenNetCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

__all__ = ["AlignnCalculator",
           "AlphaNetCalculator",
           "CHGNetCalculator",
           "DeePMDCalculator",
           "DiveNetCalculator",
           "EqV2Calculator",
           "eSENCalculator",
           "GPTFFCalculator",
           "M3GNetCalculator",
           "GraceCalculator",
           "MACECalculator",
           "MatterSimCalculator",
           "MEGNetCalculator",
           "NewtonNetCalculator",
           "ORBCalculator",
           "PosEGNNCalculator",
           "SevenNetCalculator"]
