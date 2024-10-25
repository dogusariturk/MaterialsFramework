"""
This module provides a class to generate structures
for stacking fault energy calculations using the ANNNI method.

The `ANNNIStackingFaultTransformation` class facilitates the generation of crystal structures
displaced for generalized stacking fault calculations using the ANNNI model.
It allows users to create FCC, HCP, and DHCP supercells, which are required for stacking fault
energy computations in various crystal systems.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pymatgen.core import Composition

from materialsframework.transformations.special_quasirandom_structures import SqsgenTransformation

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class ANNNIStackingFaultTransformation:
    """
    A class used to generate structures for stacking fault energy calculations using the ANNNI model.

    The `ANNNIStackingFaultTransformation` class provides methods to generate displaced structures
    in FCC, HCP, and DHCP crystal systems for generalized stacking fault energy calculations. This class
    supports customization of supercell sizes and shell weights to allow users to control the structure generation process.
    """

    def __init__(
            self,
            sqs_transformation: SqsgenTransformation | None = None
    ) -> None:
        """
        Initializes the `ANNNIStackingFaultTransformation` object.

        Args:
            sqs_transformation (SqsgenTransformation | None): An optional SQS transformation object. If not provided,
                                                                  a new instance will be created when needed.
        """
        self._sqs_transformation = sqs_transformation

        self.structures: dict[str, Structure] = {}

    def apply_transformation(
            self,
            composition: Composition | str,
            fcc_supercell_size: tuple[int, int, int] = (5, 5, 5),
            hcp_supercell_size: tuple[int, int, int] = (5, 5, 5),
            dhcp_supercell_size: tuple[int, int, int] = (5, 5, 5),
            fcc_shell_weights: dict[int, float] | None = None,
            hcp_shell_weights: dict[int, float] | None = None,
            dhcp_shell_weights: dict[int, float] | None = None
    ) -> None:
        """
        Applies the transformation to generate ANNNI stacking fault structures.

        This method generates FCC, HCP, and DHCP supercell structures based on the provided composition,
        supercell sizes, and optional shell weights, and stores the generated structures in the `structures` attribute.

        Args:
            composition (Composition | str): The composition of the supercell, either as a string or a `Composition` object.
            fcc_supercell_size (tuple[int, int, int], optional): The size of the FCC supercell. Defaults to (5, 5, 5).
            hcp_supercell_size (tuple[int, int, int], optional): The size of the HCP supercell. Defaults to (5, 5, 5).
            dhcp_supercell_size (tuple[int, int, int], optional): The size of the DHCP supercell. Defaults to (5, 5, 5).
            fcc_shell_weights (dict[int, float], optional): Shell weights for generating the FCC supercell. Defaults to None.
            hcp_shell_weights (dict[int, float], optional): Shell weights for generating the HCP supercell. Defaults to None.
            dhcp_shell_weights (dict[int, float], optional): Shell weights for generating the DHCP supercell. Defaults to None.

        Note:
            The generated structures are stored in the `structures` dictionary under the keys "fcc", "hcp", and "dhcp".
        """
        composition = Composition(composition) if isinstance(composition, str) else composition

        fcc = self.sqs_transformation.generate(composition=composition,
                                               crystal_structure="fcc_prim",
                                               supercell_size=fcc_supercell_size,
                                               shell_weights=fcc_shell_weights)
        self.structures["fcc"] = fcc["structure"]

        hcp = self.sqs_transformation.generate(composition=composition,
                                               crystal_structure="hcp",
                                               supercell_size=hcp_supercell_size,
                                               shell_weights=hcp_shell_weights)
        self.structures["hcp"] = hcp["structure"]

        dhcp = self._sqs_transformation.generate(composition=composition,
                                                 crystal_structure="dhcp",
                                                 supercell_size=dhcp_supercell_size,
                                                 shell_weights=dhcp_shell_weights)
        self.structures["dhcp"] = dhcp["structure"]

    @property
    def sqs_transformation(self) -> SqsgenTransformation:
        """
        The SqsgenTransformation object used to generate SQS structures.

        If the `sqs_transformation` instance is not already created, this property initializes a new
        `SqsgenTransformation` instance and returns it. Otherwise, it returns the existing instance.

        Returns:
            SqsgenTransformation: The SqsgenTransformation instance.
        """
        if self._sqs_transformation is None:
            self._sqs_transformation = SqsgenTransformation()
        return self._sqs_transformation
