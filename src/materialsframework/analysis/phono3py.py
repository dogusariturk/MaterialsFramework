"""
This module provides a class to calculate phonon properties of a structure using Phono3py.

The `Phono3pyAnalyzer` class facilitates phonon property calculations, including thermal
conductivity, using Phono3py. It generates displaced structures, computes forces using
the provided calculator, and calculates the thermal conductivity using the RTA or LBTE methods.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from materialsframework.calculators.m3gnet import M3GNetCalculator
from materialsframework.transformations.phono3py import Phono3pyDisplacementTransformation

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from phono3py.conductivity.direct_solution import ConductivityLBTE
    from phono3py.conductivity.rta import ConductivityRTA
    from pymatgen.core import Structure
    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class Phono3pyAnalyzer:
    """
    A class used to calculate phonon properties using Phono3py.

    The `Phono3pyAnalyzer` class provides methods to compute phonon properties of a given structure,
    including thermal conductivity using the Relaxation Time Approximation (RTA) or the Linearized
    Boltzmann Transport Equation (LBTE) methods. This is achieved by generating displaced supercells,
    calculating forces using the provided calculator, and performing phonon property calculations.
    """

    def __init__(
            self,
            calculator: BaseCalculator | None = None,
            phono3py_transformation: Phono3pyDisplacementTransformation | None = None
    ) -> None:
        """
        Initializes the `Phono3pyAnalyzer` object.

        Args:
            calculator (BaseCalculator, optional): The calculator used to compute forces and energies.
                                                   Defaults to `M3GNetCalculator` if not provided.
            phono3py_transformation (Phono3pyDisplacementTransformation, optional): The transformation object used
                                                                                   to generate displaced structures.
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
            supercell_matrix: list | None = None,
            primitive_matrix: list | None = None,
            phonon_supercell_matrix: list | None = None,
            mesh: ArrayLike | float | None = None,
            is_lbte: bool = False,
            t_min: float | None = 0,
            t_max: float | None = 1000,
            t_step: float | None = 10,
            log_level: int = 0
    ) -> dict[str, ConductivityRTA | ConductivityLBTE]:
        """
        Calculates the phonon properties of the given structure, including thermal conductivity.

        This method generates displaced supercells using Phono3py, calculates the forces using the provided calculator,
        and computes thermal conductivity based on the chosen method (RTA or LBTE).

        Args:
            structure (Structure): The structure to calculate phonon properties for.
            is_relaxed (bool, optional): Whether the input structure is already relaxed. Defaults to False.
            distance (float, optional): The distance to displace atoms for force calculations. Defaults to 0.01.
            supercell_matrix (list, optional): The supercell matrix for generating supercells. Defaults to None.
            primitive_matrix (list, optional): The primitive matrix for generating the primitive cell. Defaults to None.
            phonon_supercell_matrix (list, optional): The supercell matrix for phonon calculations. Defaults to None.
            mesh (ArrayLike | float, optional): The mesh numbers for phonon calculations. Defaults to [20, 20, 20].
            is_lbte (bool, optional): Whether to use the Linearized Boltzmann Transport Equation (LBTE). Defaults to False.
            t_min (float, optional): The minimum temperature for thermal conductivity calculations. Defaults to 0.
            t_max (float, optional): The maximum temperature for thermal conductivity calculations. Defaults to 1000.
            t_step (float, optional): The step size for temperature increments. Defaults to 10.
            log_level (int, optional): The log level for the calculations. Defaults to 0.

        Returns:
            dict[str, ConductivityRTA | ConductivityLBTE]: A dictionary containing the calculated thermal conductivity.
        """
        if "forces" not in self.calculator.AVAILABLE_PROPERTIES:
            raise ValueError("The calculator object must have the 'forces' property implemented.")

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
        self.thermal_conductivity: ConductivityRTA | ConductivityLBTE = self.phonon.thermal_conductivity

        return {
                "thermal_conductivity": self.thermal_conductivity
        }

    @property
    def calculator(self) -> BaseCalculator:
        """
        Returns the calculator used for energy and force calculations.

        If the calculator instance is not already initialized, this method creates a new `M3GNetCalculator` instance.

        Returns:
            BaseCalculator: The calculator object used for force and energy calculations.
        """
        if self._calculator is None:
            self._calculator = M3GNetCalculator()
        return self._calculator

    @property
    def phono3py_transformation(self) -> Phono3pyDisplacementTransformation:
        """
        Returns the Phono3py transformation object used to generate displaced structures.

        If the transformation instance is not already initialized, this method creates a new `Phono3pyDisplacementTransformation` instance.

        Returns:
            Phono3pyDisplacementTransformation: The transformation object used for phonon property calculations.
        """
        if self._phono3py_transformation is None:
            self._phono3py_transformation = Phono3pyDisplacementTransformation()
        return self._phono3py_transformation

    def _produce_force_constants(self) -> None:
        """
        Produces the force constants using the forces calculated from the calculator.

        This method calculates the forces on the displaced atoms using the provided calculator and then
        generates the second- and third-order force constants required for phonon calculations.
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
