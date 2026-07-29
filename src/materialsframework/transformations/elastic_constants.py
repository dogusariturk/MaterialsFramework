"""Generates distorted structures for elastic constant calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymatgen.core import Structure

from materialsframework.tools import elastic
from materialsframework.utils import to_atoms

if TYPE_CHECKING:
    from ase import Atoms

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class ElasticConstantsDeformationTransformation:
    """Generates deformed structures for elastic constant calculations."""

    def __init__(self, num_deform: int = 5, max_deform: float = 2) -> None:
        """Initializes the `ElasticConstantsDeformationTransformation` object.

        Args:
            num_deform (int, optional): The number of deformations to apply. Defaults to 5.
            max_deform (float, optional): The maximum deformation size in percent and degrees. Defaults to 2%.
        """
        self.num_deform = num_deform
        self.max_deform = max_deform

    def apply_transformation(
        self,
        structure: Structure | Atoms,
    ) -> list[Atoms]:
        """Applies the deformation transformation to the given structure and generates distorted structures.

        Args:
            structure (Structure | Atoms): The structure to apply the deformation transformation.

        Returns:
            list[Atoms]: The distorted structures generated from the deformation transformation.
        """
        if isinstance(structure, Structure):
            structure = to_atoms(structure)

        return elastic.get_elementary_deformations(cryst=structure, n=self.num_deform, d=self.max_deform)
