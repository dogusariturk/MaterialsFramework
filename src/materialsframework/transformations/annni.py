"""Generates structures for stacking fault energy calculations using the ANNNI method.

Builds FCC, HCP, and DHCP supercells displaced for generalized stacking fault energy
computations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymatgen.core import Composition

from materialsframework.tools.sqsgen import SqsGenerator
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class ANNNIStackingFaultTransformation:
    """Generates displaced structures for stacking fault energy calculations using the ANNNI model.

    Supports FCC, HCP, and DHCP crystal systems, with configurable supercell sizes and shell
    weights.
    """

    def __init__(self, sqs_gen: SqsGenerator | None = None) -> None:
        """Initializes the `ANNNIStackingFaultTransformation` object.

        Args:
            sqs_gen (SqsGenerator | None): An optional SQS generator object. If not provided,
                                                                  a new instance will be created when needed.
        """
        self._sqs_gen = sqs_gen

    def apply_transformation(
        self,
        composition: Composition | str,
        fcc_supercell_size: tuple[int, int, int] = (5, 5, 5),
        hcp_supercell_size: tuple[int, int, int] = (5, 5, 5),
        dhcp_supercell_size: tuple[int, int, int] = (5, 5, 5),
        fcc_shell_weights: dict[int, float] | None = None,
        hcp_shell_weights: dict[int, float] | None = None,
        dhcp_shell_weights: dict[int, float] | None = None,
    ) -> dict[str, Structure]:
        """Generate FCC, HCP, and DHCP supercell structures for the given composition.

        Args:
            composition (Composition | str): The composition of the supercell, either as a string or a `Composition` object.
            fcc_supercell_size (tuple[int, int, int], optional): The size of the FCC supercell. Defaults to (5, 5, 5).
            hcp_supercell_size (tuple[int, int, int], optional): The size of the HCP supercell. Defaults to (5, 5, 5).
            dhcp_supercell_size (tuple[int, int, int], optional): The size of the DHCP supercell. Defaults to (5, 5, 5).
            fcc_shell_weights (dict[int, float], optional): Shell weights for generating the FCC supercell. Defaults to None.
            hcp_shell_weights (dict[int, float], optional): Shell weights for generating the HCP supercell. Defaults to None.
            dhcp_shell_weights (dict[int, float], optional): Shell weights for generating the DHCP supercell. Defaults to None.

        Returns:
            dict[str, Structure]: Dictionary of the generated structures, keyed by "fcc", "hcp", and "dhcp".
        """
        composition = Composition(composition) if isinstance(composition, str) else composition

        structures: dict[str, Structure] = {
            "fcc": self.sqs_gen.generate(
                composition=composition,
                crystal_structure="fcc_prim",
                supercell_size=fcc_supercell_size,
                shell_weights=fcc_shell_weights,
            )["structure"],
            "hcp": self.sqs_gen.generate(
                composition=composition,
                crystal_structure="hcp",
                supercell_size=hcp_supercell_size,
                shell_weights=hcp_shell_weights,
            )["structure"],
            "dhcp": self.sqs_gen.generate(
                composition=composition,
                crystal_structure="dhcp",
                supercell_size=dhcp_supercell_size,
                shell_weights=dhcp_shell_weights,
            )["structure"],
        }

        return structures

    @lazy_property("_sqs_gen")
    def sqs_gen(self) -> SqsGenerator:
        """The SqsGenerator used to generate SQS structures.

        Lazily creates a `SqsGenerator` instance on first access.

        Returns:
            SqsGenerator: The SqsGenerator instance.
        """
        return SqsGenerator()
