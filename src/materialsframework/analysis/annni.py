"""This module provides a class to perform the second-order ANNNI formulae on a composition to calculate intrinsic and extrinsic stacking fault energies.

The `ANNNIStackingFaultAnalyzer` class derives the intrinsic and extrinsic stacking fault energies
(ISFE and ESFE) from the energy differences between FCC, HCP, and DHCP structures, using the
second-order ANNNI (Axial Next-Nearest Neighbor Ising) model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from materialsframework.analysis.base import BaseAnalyzer
from materialsframework.analysis.utils import require_properties
from materialsframework.transformations.annni import ANNNIStackingFaultTransformation
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from pymatgen.core import Composition

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class ANNNIStackingFaultAnalyzer(BaseAnalyzer):
    """A class used to calculate intrinsic and extrinsic stacking fault energies using the ANNNI model.

    The `ANNNIStackingFaultAnalyzer` class compares the potential energies of FCC, HCP, and DHCP
    structures to compute the intrinsic and extrinsic stacking fault energies (ISFE and ESFE) per the
    second-order ANNNI formulae.
    """

    def __init__(
        self,
        calculator: BaseCalculator | None = None,
        annni_transformation: ANNNIStackingFaultTransformation | None = None,
    ) -> None:
        """Initializes the `ANNNIStackingFaultAnalyzer` object.

        Args:
            calculator (BaseCalculator | None, optional): The calculator object used for relaxation and potential energy calculations.
            annni_transformation (ANNNIStackingFaultTransformation | None, optional): The transformation object used
                to generate stacking fault structures. If not provided, a default instance is initialized.
        """
        super().__init__(calculator)
        self._annni_transformation = annni_transformation

    @require_properties("energy")
    def calculate(self, composition: Composition | str) -> dict[str, float]:
        """Calculates intrinsic and extrinsic stacking fault energies (ISFE and ESFE) using the second-order ANNNI formulae.

        The energy differences between FCC, HCP, and DHCP structures are normalized by the area of the
        FCC unit cell.

        Args:
            composition (Composition | str): The composition of the supercell, either as a `Composition` object or as
                a string.

        Returns:
            dict[str, float]: Dictionary with keys:
                - ``isfe``: Intrinsic stacking fault energy (eV/Å²).
                - ``esfe``: Extrinsic stacking fault energy (eV/Å²).

        Raises:
            ValueError: If the calculator object does not have the 'energy' property implemented.
        """
        structures = self.annni_transformation.apply_transformation(composition=composition)

        fcc_struct = structures["fcc"]
        fcc_result = self.calculator.relax(fcc_struct)
        fcc_energy = fcc_result["energy"] / fcc_result["final_structure"].num_sites
        fcc_vol_per_atom = fcc_result["final_structure"].volume / fcc_result["final_structure"].num_sites
        a_conv = (4 * fcc_vol_per_atom) ** (1 / 3)
        a_fcc = np.sqrt(3) / 4 * a_conv**2

        hcp_struct = structures["hcp"]
        hcp_struct = hcp_struct.scale_lattice(fcc_vol_per_atom * hcp_struct.num_sites)
        hcp_result = self.calculator.calculate(hcp_struct)
        hcp_energy = hcp_result["energy"] / hcp_struct.num_sites

        dhcp_struct = structures["dhcp"]
        dhcp_struct = dhcp_struct.scale_lattice(fcc_vol_per_atom * dhcp_struct.num_sites)
        dhcp_result = self.calculator.calculate(dhcp_struct)
        dhcp_energy = dhcp_result["energy"] / dhcp_struct.num_sites

        return {
            "isfe": (hcp_energy + (2 * dhcp_energy) - (3 * fcc_energy)) / a_fcc,
            "esfe": (4 * (dhcp_energy - fcc_energy)) / a_fcc,
        }

    @lazy_property("_annni_transformation")
    def annni_transformation(self) -> ANNNIStackingFaultTransformation:
        """Returns the ANNNI stacking fault transformation object used to generate stacking fault structures.

        Returns:
            ANNNIStackingFaultTransformation: The transformation object used to generate stacking fault structures.
        """
        return ANNNIStackingFaultTransformation()
