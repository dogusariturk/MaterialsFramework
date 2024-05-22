"""
This module provides a class to perform the second-order ANNNI formulae on a composition
to calculate intrinsic and extrinsic stacking fault energies.
"""
from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import numpy as np

from materialsframework.calculators import M3GNetRelaxer
from materialsframework.transformations import ANNNIStackingFaultTransformation

if TYPE_CHECKING:
    from pymatgen.core import Composition
    from materialsframework.calculators.typing import Relaxer

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
            annni_transformation: Optional[ANNNIStackingFaultTransformation] = None
    ) -> None:
        """
        Initializes the ANNNIStackingFaultAnalyzer object.

        Parameters:
            relaxer (Optional[Relaxer]): The Relaxer object to use for relaxation. Defaults to None.
            annni_transformation (Optional[ANNNIStackingFaultTransformation]): The ANNNI stacking fault transformation object.
        """
        self._relaxer = relaxer
        self._annni_transformation = annni_transformation

    def calculate(self, composition: Composition):
        """
        Calculates the intrinsic and extrinsic stacking fault energies using the second-order ANNNI formulae.

        Args:
            composition (Composition): The composition of the supercell.

        Returns:
            dict: A dictionary containing the intrinsic stacking fault energy (isfe)
                  and extrinsic stacking fault energy (esfe).
        """
        structures = self.annni_transformation.apply_transformation(composition=composition).structures

        fcc_result = self.relaxer.relax(structures["fcc"])
        fcc_energy = fcc_result["energy"]
        fcc_volume = fcc_result["final_structure"].volume
        a_fcc = np.sqrt(3) / 4 * fcc_result["final_structure"].lattice.a ** 2

        hcp_energy = self.relaxer.relax(structures["hcp"].scale_lattice(fcc_volume))["energy"]
        dhcp_energy = self.relaxer.relax(structures["dhcp"].scale_lattice(fcc_volume))["energy"]

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
    def annni_transformation(self) -> ANNNIStackingFaultTransformation:
        """
        Gets the ANNNI stacking fault transformation object.

        Returns:
            ANNNIStackingFaultTransformation: The ANNNI stacking fault transformation object.
        """
        if self._annni_transformation is None:
            self._annni_transformation = ANNNIStackingFaultTransformation()
        return ANNNIStackingFaultTransformation()
