"""
This module provides a class to calculate phonon properties of a structure using Phono3py.
"""
from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING, Union

from materialsframework.calculators.m3gnet import M3GNetCalculator
from materialsframework.transformations.phono3py import Phono3pyDisplacementTransformation

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from phono3py.conductivity.direct_solution import ConductivityLBTE
    from phono3py.conductivity.rta import ConductivityRTA
    from pymatgen.core import Structure
    from materialsframework.tools.typing import Calculator

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class Phono3pyAnalyzer:
    """
    A class used to represent a Phono3py Analyzer.

    This class provides methods to calculate phonon properties of a structure.
    """

    def __init__(
            self,
            calculator: Optional[Calculator] = None,
            phono3py_transformation: Optional[Phono3pyDisplacementTransformation] = None
    ) -> None:
        """
        Initializes the Phono3pyAnalyzer.

        Args:
            calculator (Calculator): The calculator to use for calculating energies and forces.
            phono3py_transformation (Phono3pyDisplacementTransformation): The Phono3pyDisplacementTransformation object.
        """
        self._calculator = calculator
        self._phono3py_transformation = phono3py_transformation

        self.phonon = None
        self.thermal_conductivity = None

    def calculate(
            self,
            structure: Structure,
            is_relaxed: bool = False,
            distance: float = 0.01,
            supercell_matrix: Optional[list] = None,
            primitive_matrix: Optional[list] = None,
            phonon_supercell_matrix: Optional[list] = None,
            mesh: Optional[ArrayLike | float] = None,
            is_lbte: bool = False,
            t_min: Optional[float] = 0,
            t_max: Optional[float] = 1000,
            t_step: Optional[float] = 10,
            log_level: int = 0
    ) -> dict:
        """
        Calculates the phonon properties of the given structure.

        Args:
            structure (Structure): The structure to calculate phonon properties.
            is_relaxed (bool): Whether the structure is relaxed. Defaults to False.
            distance (float): The distance to displace atoms for forces. Defaults to 0.01.
            supercell_matrix (list): The supercell matrix. Defaults to None.
            primitive_matrix (list): The primitive matrix. Defaults to None.
            phonon_supercell_matrix (list): The phonon supercell matrix. Defaults to None.
            mesh (ArrayLike | float): The mesh numbers for phonon calculations. Defaults to None.
            is_lbte (bool): Whether to use LBTE for thermal conductivity. Defaults to False.
            t_min (float): The minimum temperature for thermal conductivity. Defaults to 0.
            t_max (float): The maximum temperature for thermal conductivity. Defaults to 1000.
            t_step (float): The temperature step for thermal conductivity. Defaults to 10.
            log_level (int): The log level for the calculations. Defaults to 0.

        Returns:
            dict: A dictionary containing the calculated thermal conductivity.
        """
        mesh = mesh or [20, 20, 20]

        self.phono3py_transformation.apply_transformation(structure=structure,
                                                          distance=distance,
                                                          supercell_matrix=supercell_matrix,
                                                          primitive_matrix=primitive_matrix,
                                                          is_relaxed=is_relaxed,
                                                          phonon_supercell_matrix=phonon_supercell_matrix,
                                                          log_level=log_level)

        self.phonon = self.phono3py_transformation.phonon
        self._produce_force_constants()

        self.phonon.mesh_numbers = mesh
        self.phonon.init_phph_interaction()

        # Thermal Conductivity
        self.phonon.run_thermal_conductivity(is_LBTE=is_lbte,
                                             temperatures=range(t_min, t_max + 1, t_step))
        self.thermal_conductivity: Union[ConductivityRTA, ConductivityLBTE] = self.phonon.thermal_conductivity

        return {
                "thermal_conductivity": self.thermal_conductivity
        }

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
    def phono3py_transformation(self) -> Phono3pyDisplacementTransformation:
        """
        Gets the Phono3py transformation object.

        Returns:
            Phono3pyDisplacementTransformation: The Phono3py transformation object.
        """
        if self._phono3py_transformation is None:
            self._phono3py_transformation = Phono3pyDisplacementTransformation()
        return self._phono3py_transformation

    def _produce_force_constants(self) -> None:
        """
        Produces the force constants using the forces calculated from the calculator.
        """
        if self.phonon is None:
            raise RuntimeError("phono3py_transformation has to be called before trying to produce force constants.")

        forces = [self.calculator.calculate(displaced_structure)["forces"].tolist()
                  for displaced_structure in
                  self.phono3py_transformation.supercells_with_displacements]
        self.phonon.forces = forces

        phonon_forces = [self.calculator.calculate(displaced_structure)["forces"].tolist()
                         for displaced_structure in
                         self.phono3py_transformation.phonon_supercells_with_displacements]
        self.phonon.phonon_forces = phonon_forces

        self.phonon.produce_fc3()
        self.phonon.produce_fc2()
