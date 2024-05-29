"""
This module provides a class to generate structures
for stacking fault energy calculations using the ANNNI method.
"""
from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING, Union

from pymatgen.core import Composition

from materialsframework.transformations.special_quasirandom_structures import SqsgenTransformation

if TYPE_CHECKING:
    from pymatgen.core import Structure

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class ANNNIStackingFaultTransformation:
    """
    A class used to represent a ANNNIStackingFaultTransformation object.

    This class provides methods to generate displaced structures
    for generalized stacking fault calculations.
    """

    def __init__(
            self,
            sqs_transformation: Optional[SqsgenTransformation] = None
    ) -> None:
        """
        Initializes the ANNNIStackingFaultTransformation object.

        Args:
            sqs_transformation (Optional[SqsgenTransformation]): The SQS transformation object. Default is SqsgenTransformation.
        """
        self._sqs_transformation = sqs_transformation

        self.structures: dict[str, Structure] = {}

    def apply_transformation(
            self,
            composition: Union[Composition, str],
            fcc_supercell_size: int = (5, 5, 5),
            hcp_supercell_size: int = (5, 5, 5),
            dhcp_supercell_size: int = (5, 5, 5),
            fcc_shell_weights: Optional[dict[int, float]] = None,
            hcp_shell_weights: Optional[dict[int, float]] = None,
            dhcp_shell_weights: Optional[dict[int, float]] = None
    ) -> None:
        """
        Applies the transformation to generate ANNNI stacking fault structures.

        Args:
            composition (Union[Composition,str]): The composition of the supercell.
            fcc_supercell_size (int): The size of the FCC supercell. Default is (5, 5, 5).
            hcp_supercell_size (int): The size of the HCP supercell. Default is (5, 5, 5).
            dhcp_supercell_size (int): The size of the DHCP supercell. Default is (5, 5, 5).
            fcc_shell_weights (dict[int, float]): The shell weights for the FCC supercell. Default is None.
            hcp_shell_weights (dict[int, float]): The shell weights for the HCP supercell. Default is None.
            dhcp_shell_weights (dict[int, float]): The shell weights for the DHCP supercell. Default is None.
        """
        composition = Composition(composition) if isinstance(composition, str) else composition

        fcc = self.sqs_transformation.generate(composition=composition,
                                               crystal_structure='fcc_prim',
                                               supercell_size=fcc_supercell_size,
                                               shell_weights=fcc_shell_weights)
        self.structures['fcc'] = fcc['structure']

        hcp = self.sqs_transformation.generate(composition=composition,
                                               crystal_structure='hcp',
                                               supercell_size=hcp_supercell_size,
                                               shell_weights=hcp_shell_weights)
        self.structures['hcp'] = hcp['structure']

        dhcp = self._sqs_transformation.generate(composition=composition,
                                                 crystal_structure='dhcp',
                                                 supercell_size=dhcp_supercell_size,
                                                 shell_weights=dhcp_shell_weights)
        self.structures['dhcp'] = dhcp['structure']

    @property
    def sqs_transformation(self) -> SqsgenTransformation:
        """
        The SqsgenTransformation object used to generate SQS structures.
        Returns the SqsgenTransformation instance.

        If the sqs_transformation instance is not already created, it creates a new SqsgenTransformation instance
        and returns it. Otherwise, it returns the existing sqs_transformation instance.

        Returns:
            SqsgenTransformation: The SqsgenTransformation instance.
        """
        if self._sqs_transformation is None:
            self._sqs_transformation = SqsgenTransformation()
        return self._sqs_transformation
