"""
This module provides a class to generate distorted structures for elastic constant calculations.

The `ElasticConstantsDeformationTransformation` class facilitates the generation of distorted
structures required for the calculation of elastic constants.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from elastic import elastic

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from ase import Atoms

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class ElasticConstantsDeformationTransformation:
    """
    A class used to generate deformed structures for elastic constant calculations.

    The `ElasticConstantsDeformationTransformation` class provides methods to generate distorted
    structures for the calculation of elastic constants.
    """

    def __init__(
            self,
            num_deform: int = 5,
            max_deform: float = 2
    ) -> None:
        """
        Initializes the `ElasticConstantsDeformationTransformation` object.

        Args:
            num_deform (int, optional): The number of deformations to apply. Defaults to 5.
            max_deform (float, optional): The maximum deformation size in percent and degrees. Defaults to 2%.
        """
        self.num_deform = num_deform
        self.max_deform = max_deform

        self.distorted_structures: [Atoms] = []

    def apply_transformation(
            self,
            structure: Structure,
    ) -> None:
        """
        Applies the deformation transformation to the given structure and generates distorted structures.

        Args:
            structure (Structure): The structure to apply the deformation transformation.
        """
        ase_atoms = structure.to_ase_atoms()
        self.distorted_structures = elastic.get_elementary_deformations(
                cryst=ase_atoms,
                n=self.num_deform,
                d=self.max_deform
        )
