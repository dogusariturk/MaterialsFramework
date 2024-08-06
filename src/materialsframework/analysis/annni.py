"""
This module provides a class to perform the second-order ANNNI formulae on a composition
to calculate intrinsic and extrinsic stacking fault energies.
"""
from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING, Union

import numpy as np

from materialsframework.calculators.m3gnet import M3GNetCalculator, M3GNetRelaxer
from materialsframework.transformations.annni import ANNNIStackingFaultTransformation

if TYPE_CHECKING:
    from pymatgen.core import Composition
    from materialsframework.tools.typing import Calculator, Relaxer

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class ANNNIStackingFaultAnalyzer:
    """
    A class used to represent an ANNNIStackingFaultAnalyzer object.

    This class provides methods to calculate intrinsic and extrinsic stacking fault energies
    using the second-order ANNNI formulae.
    """

    def __init__(
            self,
            relaxer: Optional[Relaxer] = None,
            calculator: Optional[Calculator] = None,
            annni_transformation: Optional[ANNNIStackingFaultTransformation] = None
    ) -> None:
        """
        Initializes the ANNNIStackingFaultAnalyzer object.

        Parameters:
            relaxer (Optional[Relaxer]): The Relaxer object to use for relaxation. Defaults to M3GNetRelaxer.
            calculator (Optional[Calculator]): The Calculator object to use for calculating potential energies. Defaults to M3GNetCalculator.
            annni_transformation (Optional[ANNNIStackingFaultTransformation]): The ANNNI stacking fault transformation object.
        """
        self._relaxer = relaxer
        self._calculator = calculator
        self._annni_transformation = annni_transformation

    def calculate(self, composition: Union[Composition, str]) -> dict:
        """
        Calculates the intrinsic and extrinsic stacking fault energies using the second-order ANNNI formulae.

        Args:
            composition (Union[Composition,str]): The composition of the supercell.

        Returns:
            dict: A dictionary containing the intrinsic stacking fault energy (isfe)
                  and extrinsic stacking fault energy (esfe).
        """
        self.annni_transformation.apply_transformation(composition=composition)

        fcc_struct = self.annni_transformation.structures["fcc"]
        fcc_result = self.relaxer.relax(fcc_struct)
        fcc_energy = fcc_result["energy"] / fcc_result["final_structure"].num_sites
        fcc_volume = fcc_result["final_structure"].volume
        a_fcc = np.sqrt(3) / 4 * (fcc_result["final_structure"].lattice.matrix[0][1] * 2) ** 2

        hcp_struct = self.annni_transformation.structures["hcp"].scale_lattice(fcc_volume)
        hcp_result = self.calculator.calculate(hcp_struct)
        hcp_energy = hcp_result["potential_energy"] / hcp_struct.num_sites

        dhcp_struct = self.annni_transformation.structures["dhcp"].scale_lattice(fcc_volume)
        dhcp_result = self.calculator.calculate(dhcp_struct)
        dhcp_energy = dhcp_result["potential_energy"] / dhcp_struct.num_sites

        return {
                "isfe": (hcp_energy + (2 * dhcp_energy) - (3 * fcc_energy)) / a_fcc,
                "esfe": (4 * (dhcp_energy - fcc_energy)) / a_fcc
        }

    @property
    def relaxer(self) -> Relaxer:
        """
        Returns the Relaxer instance.

        If the relaxer instance is not already created, it creates a new M3GNetRelaxer instance
        and returns it. Otherwise, it returns the existing relaxer instance.

        Returns:
            Relaxer: The Relaxer instance.
        """
        if self._relaxer is None:
            self._relaxer = M3GNetRelaxer()
        return self._relaxer

    @property
    def calculator(self) -> Calculator:
        """
        Gets the calculator used for calculating potential energies.
        If not set, initializes a new M3GNetCalculator.

        Returns:
            Calculator: The calculator object.
        """
        if self._calculator is None:
            self._calculator = M3GNetCalculator()
        return self._calculator

    @property
    def annni_transformation(self) -> ANNNIStackingFaultTransformation:
        """
        Gets the ANNNI stacking fault transformation object.

        Returns:
            ANNNIStackingFaultTransformation: The ANNNI stacking fault transformation object.
        """
        if self._annni_transformation is None:
            self._annni_transformation = ANNNIStackingFaultTransformation()
        return self._annni_transformation
