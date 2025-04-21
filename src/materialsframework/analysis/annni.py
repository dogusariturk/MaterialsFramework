"""
This module provides a class to perform the second-order ANNNI formulae on a composition
to calculate intrinsic and extrinsic stacking fault energies.

The `ANNNIStackingFaultAnalyzer` class calculates stacking fault energies, which are essential
for understanding the stability of certain crystal structures, using the second-order ANNNI (Axial
Next-Nearest Neighbor Ising) model. The intrinsic and extrinsic stacking fault energies are derived
based on the energy differences between FCC, HCP, and DHCP structures.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from materialsframework.transformations.annni import ANNNIStackingFaultTransformation

if TYPE_CHECKING:
    from pymatgen.core import Composition
    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class ANNNIStackingFaultAnalyzer:
    """
    A class used to calculate intrinsic and extrinsic stacking fault energies using the ANNNI model.

    The `ANNNIStackingFaultAnalyzer` class provides methods to compute the intrinsic and extrinsic stacking
    fault energies (ISFE and ESFE) based on the second-order ANNNI formulae. The energies are computed by
    comparing the potential energies of FCC, HCP, and DHCP structures. These energies are important for
    understanding the material's stacking fault behavior, especially in metallic alloys.
    """

    def __init__(
            self,
            calculator: BaseCalculator | None = None,
            annni_transformation: ANNNIStackingFaultTransformation | None = None
    ) -> None:
        """
        Initializes the `ANNNIStackingFaultAnalyzer` object.

        Args:
            calculator (BaseCalculator | None, optional): The calculator object used for relaxation and potential energy calculations.
                                                            Defaults to `M3GNetCalculator`.
            annni_transformation (ANNNIStackingFaultTransformation | None, optional): The transformation object used to generate stacking
                                                                                        fault structures. If not provided, a default instance
                                                                                        is initialized.
        """
        self._calculator = calculator
        self._annni_transformation = annni_transformation

    def calculate(
            self,
            composition: Composition | str
    ) -> dict:
        """
        Calculates intrinsic and extrinsic stacking fault energies (ISFE and ESFE) using the second-order ANNNI formulae.

        This method calculates the intrinsic and extrinsic stacking fault energies based on the energy differences
        between FCC, HCP, and DHCP structures. The stacking fault energies are normalized by the area of the FCC
        unit cell. The final results are returned as a dictionary.

        Args:
            composition (Composition | str): The composition of the supercell, either as a `Composition` object
                                                   or as a string.

        Returns:
            dict: A dictionary containing the intrinsic stacking fault energy (`isfe`) and extrinsic stacking fault
                  energy (`esfe`), both normalized by the FCC unit cell area.
        """
        if "energy" not in self.calculator.AVAILABLE_PROPERTIES:
            raise ValueError("The calculator object must have the 'energy' property implemented.")

        self.annni_transformation.apply_transformation(composition=composition)

        fcc_struct = self.annni_transformation.structures["fcc"]
        fcc_result = self.calculator.relax(fcc_struct)
        fcc_energy = fcc_result["energy"] / fcc_result["final_structure"].num_sites
        fcc_volume = fcc_result["final_structure"].volume
        a_fcc = np.sqrt(3) / 4 * (fcc_result["final_structure"].lattice.matrix[0][1] * 2) ** 2

        hcp_struct = self.annni_transformation.structures["hcp"].scale_lattice(fcc_volume)
        hcp_result = self.calculator.calculate(hcp_struct)
        hcp_energy = hcp_result["energy"] / hcp_struct.num_sites

        dhcp_struct = self.annni_transformation.structures["dhcp"].scale_lattice(fcc_volume)
        dhcp_result = self.calculator.calculate(dhcp_struct)
        dhcp_energy = dhcp_result["energy"] / dhcp_struct.num_sites

        return {
                "isfe": (hcp_energy + (2 * dhcp_energy) - (3 * fcc_energy)) / a_fcc,
                "esfe": (4 * (dhcp_energy - fcc_energy)) / a_fcc
        }

    @property
    def calculator(self) -> BaseCalculator:
        """
        Returns the calculator instance used for energy calculations.

        If the calculator instance is not already initialized, this method creates a new `M3GNetCalculator` instance.

        Returns:
            BaseCalculator: The calculator object used for relaxation and energy calculations.
        """
        if self._calculator is None:
            from materialsframework.calculators.m3gnet import M3GNetCalculator
            self._calculator = M3GNetCalculator()
        return self._calculator

    @property
    def annni_transformation(self) -> ANNNIStackingFaultTransformation:
        """
        Returns the ANNNI stacking fault transformation object used to generate stacking fault structures.

        If the transformation instance is not already initialized, this method creates a new `ANNNIStackingFaultTransformation` instance.

        Returns:
            ANNNIStackingFaultTransformation: The transformation object used to generate stacking fault structures.
        """
        if self._annni_transformation is None:
            self._annni_transformation = ANNNIStackingFaultTransformation()
        return self._annni_transformation
