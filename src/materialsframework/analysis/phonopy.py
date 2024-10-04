"""
This module provides a class to calculate phonon properties of a structure using Phonopy.

The `PhonopyAnalyzer` class facilitates phonon property calculations, including the total density of states (DOS),
projected DOS (PDOS), and thermal properties of a structure. It leverages Phonopy for calculating these properties
and utilizes transformations to generate displaced structures and force constants.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from materialsframework.calculators.m3gnet import M3GNetCalculator
from materialsframework.transformations.phonopy import PhonopyDisplacementTransformation

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure
    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class PhonopyAnalyzer:
    """
    A class used to calculate phonon properties using Phonopy.

    The `PhonopyAnalyzer` class provides methods to compute phonon properties such as the total
    density of states (DOS), projected DOS (PDOS), and thermal properties of a given structure.
    It generates displaced supercells, calculates forces using a specified calculator, and computes
    the required phonon properties.
    """

    def __init__(
            self,
            calculator: BaseCalculator | None = None,
            phonopy_transformation: PhonopyDisplacementTransformation | None = None
    ) -> None:
        """
        Initializes the `PhonopyAnalyzer` object.

        Args:
            calculator (BaseCalculator, optional): The calculator used to compute forces and energies.
                                                   Defaults to `M3GNetCalculator`.
            phonopy_transformation (PhonopyDisplacementTransformation, optional): The transformation object used
                                                                                  to generate displaced supercells.
        """
        self._calculator = calculator
        self._phonopy_transformation = phonopy_transformation

        self.phonon = None
        self.total_dos = None
        self.projected_dos = None
        self.thermal_properties = None

    def calculate(
            self,
            structure: Structure,
            is_relaxed: bool = False,
            distance: float = 0.01,
            supercell_matrix: list | None = None,
            primitive_matrix: list | None = None,
            mesh: ArrayLike | float | None = None,
            pdos_mesh: ArrayLike | float | None = None,
            sigma: float | None = None,
            freq_min: float | None = None,
            freq_max: float | None = None,
            freq_pitch: float | None = None,
            t_min: float | None = 0,
            t_max: float | None = 1000,
            t_step: float | None = 10,
            log_level: int = 0
    ) -> dict[str, dict]:
        """
        Calculates the phonon properties of the given structure.

        This method generates displaced supercells using Phonopy, calculates the forces using the provided calculator,
        and computes the total density of states (DOS), projected DOS (PDOS), and thermal properties.

        Args:
            structure (Structure): The structure to calculate phonon properties for.
            is_relaxed (bool, optional): Whether the input structure is already relaxed. Defaults to False.
            distance (float, optional): The distance to displace atoms for force calculations. Defaults to 0.01.
            supercell_matrix (list, optional): The supercell matrix for generating supercells. Defaults to None.
            primitive_matrix (list, optional): The primitive matrix for generating the primitive cell. Defaults to None.
            mesh (ArrayLike | float, optional): The mesh numbers for phonon calculations. Defaults to [20, 20, 20].
            pdos_mesh (ArrayLike | float, optional): The mesh numbers for projected DOS calculations. Defaults to [10, 10, 10].
            sigma (float, optional): The smearing width for the total DOS calculation. Defaults to None.
            freq_min (float, optional): The minimum frequency for the total DOS calculation. Defaults to None.
            freq_max (float, optional): The maximum frequency for the total DOS calculation. Defaults to None.
            freq_pitch (float, optional): The interval of frequencies for the total DOS calculation. Defaults to None.
            t_min (float, optional): The minimum temperature for thermal property calculations. Defaults to 0.
            t_max (float, optional): The maximum temperature for thermal property calculations. Defaults to 1000.
            t_step (float, optional): The step size for temperature increments. Defaults to 10.
            log_level (int, optional): The log level for the phonon calculations. Defaults to 0.

        Returns:
            dict[str, dict]: A dictionary containing the calculated total DOS, thermal properties, and projected DOS.
        """
        if "forces" not in self.calculator.AVAILABLE_PROPERTIES:
            raise ValueError("The calculator object must have the 'forces' property implemented.")

        mesh = mesh or [20, 20, 20]
        pdos_mesh = pdos_mesh or [10, 10, 10]

        self.phonopy_transformation.apply_transformation(structure=structure,
                                                         distance=distance,
                                                         supercell_matrix=supercell_matrix,
                                                         primitive_matrix=primitive_matrix,
                                                         is_relaxed=is_relaxed,
                                                         log_level=log_level)
        self.phonon = self.phonopy_transformation.phonon
        self._produce_force_constants()

        self.phonon.run_mesh(mesh=mesh)

        # DOS
        self.phonon.run_total_dos(sigma=sigma,
                                  freq_min=freq_min,
                                  freq_max=freq_max,
                                  freq_pitch=freq_pitch)
        total_dos = self.phonon.get_total_dos_dict()

        # Thermal Properties
        self.phonon.run_thermal_properties(t_min=t_min,
                                           t_max=t_max,
                                           t_step=t_step)
        thermal_properties = self.phonon.get_thermal_properties_dict()

        # PDOS
        self.phonon.run_mesh(mesh=pdos_mesh,
                             is_mesh_symmetry=False,
                             with_eigenvectors=True)
        self.phonon.run_projected_dos(use_tetrahedron_method=True)
        projected_dos = self.phonon.get_projected_dos_dict()

        return {
                "total_dos": total_dos,
                "thermal_properties": thermal_properties,
                "projected_dos": projected_dos
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
    def phonopy_transformation(self) -> PhonopyDisplacementTransformation:
        """
        Returns the Phonopy transformation object used to generate displaced structures.

        If the transformation instance is not already initialized, this method creates a new `PhonopyDisplacementTransformation` instance.

        Returns:
            PhonopyDisplacementTransformation: The transformation object used for phonon property calculations.
        """
        if self._phonopy_transformation is None:
            self._phonopy_transformation = PhonopyDisplacementTransformation()
        return self._phonopy_transformation

    def _produce_force_constants(self) -> None:
        """
        Produces the force constants using the forces calculated from the calculator.

        This method calculates the forces on the displaced atoms using the provided calculator and generates
        the force constants required for phonon calculations.
        """
        if self.phonon is None:
            raise RuntimeError("phonopy_transformation has to be called before trying to produce force constants.")

        forces = [self.calculator.calculate(displaced_structure)["forces"]
                  for displaced_structure in self.phonopy_transformation.displaced_structures]
        self.phonon.forces = forces
        self.phonon.produce_force_constants()
