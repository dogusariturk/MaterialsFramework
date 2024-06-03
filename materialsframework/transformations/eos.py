"""
This module contains the EOSTransformation class that generates deformed structures for EOS calculations.
"""
from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.calculators.typing import Relaxer

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class EOSTransformation:
    """
    A class used to represent an EOS Transformation.

    This class provides methods to generate deformed structures for EOS calculations.
    """

    def __init__(
            self,
            start: float = -0.01,
            stop: float = 0.01,
            num: int = 5,
            relaxer: Optional[Relaxer] = None
    ) -> None:
        """
        Initializes the EOSTransformation.

        Args:
            start (float): The starting strain value. Defaults to -0.01.
            stop (float): The stopping strain value. Defaults to 0.01.
            num (int): The number of strain values. Defaults to 5.
            relaxer (Optional[Relaxer]): The Relaxer object to use for relaxation. Default is M3GNetRelaxer.
        """
        self._relaxer = relaxer

        self._strains = np.linspace(start, stop, num)

        self.structures = {}

    def apply_transformation(
            self,
            undeformed_structure: Structure
    ) -> None:
        """
        Applies the transformation to generate deformed structures for the EOS calculations.

        Args:
            undeformed_structure (Structure): The undeformed structure.
        """
        initial_volume = undeformed_structure.volume

        for strain in self._strains:
            structure = undeformed_structure.copy()
            deformed_structure = structure.scale_lattice(initial_volume * (1 + strain))
            self.structures[strain] = deformed_structure
