"""
This module provides a class to calculate phonon properties of a structure using Phonopy.
"""
from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import numpy as np

from materialsframework.calculators import M3GNetCalculator
from materialsframework.transformations import PhonopyDisplacementTransformation

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure
    from materialsframework.calculators.typing import Calculator

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class PhonopyAnalyzer:
    """
    A class used to represent a Phonopy Analyzer.

    This class provides methods to calculate phonon properties of a structure.
    """

    def __init__(
            self,
            calculator: Optional[Calculator] = None,
            phonopy_transformation: Optional[PhonopyDisplacementTransformation] = None
    ) -> None:
        """
        Initializes the PhonopyAnalyzer.

        Args:
            calculator (Calculator): The calculator to use for calculating energies and forces.
            phonopy_transformation (PhonopyDisplacementTransformation): The PhonopyDisplacementTransformation object.
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
            supercell_matrix: Optional[list] = None,
            primitive_matrix: Optional[list] = None,
            mesh: Optional[ArrayLike | float] = None,
            pdos_mesh: Optional[ArrayLike | float] = None,
            sigma: Optional[float] = None,
            freq_min: Optional[float] = None,
            freq_max: Optional[float] = None,
            freq_pitch: Optional[float] = None,
            t_min: Optional[float] = 0,
            t_max: Optional[float] = 1000,
            t_step: Optional[float] = 10,
            log_level: int = 0
    ) -> None:
        """
        Calculates the phonon properties of the given structure.

        Args:
            structure (Structure): The structure to calculate the phonon properties.
            is_relaxed (bool): Whether the structure is relaxed. Defaults to False.
            distance (float): The distance to displace the atoms for the force calculations. Defaults to 0.01.
            supercell_matrix (list): The supercell matrix to use for the phonon calculations.
            primitive_matrix (list): The primitive matrix to use for the phonon calculations.
            mesh (ArrayLike | float): The mesh to use for the phonon calculations.
            pdos_mesh (ArrayLike | float): The mesh to use for the phonon calculations for PDOS.
            sigma (float): The smearing width for smearing method to use for the total DOS.
            freq_min (float): The minimum frequency at which total DOS is computed.
            freq_max (float): The maximum frequency at which total DOS is computed.
            freq_pitch (float): The interval of frequencies to use for the total DOS.
            t_min (float): The minimum temperature to use for the thermal properties.
            t_max (float): The maximum temperature to use for the thermal properties.
            t_step (float): The step to use for the thermal properties.
            log_level (int): The log level to use for the phonon calculations. Defaults to 0.
        """
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
    def phonopy_transformation(self) -> PhonopyDisplacementTransformation:
        """
        Gets the phonopy transformation object.

        Returns:
            PhonopyDisplacementTransformation: The phonopy transformation object.
        """
        if self._phonopy_transformation is None:
            self._phonopy_transformation = PhonopyDisplacementTransformation()
        return self._phonopy_transformation

    def _produce_force_constants(self) -> None:
        """
        Produces the force constants using the forces calculated from the calculator.
        """
        if self.phonon is None:
            raise RuntimeError("phonopy_transformation has to be called before trying to produce force constants.")

        forces = np.array([self.calculator.calculate(displaced_structure)["forces"]
                           for displaced_structure in self.phonopy_transformation.displaced_structures])
        self.phonon.forces = forces
        self.phonon.produce_force_constants()
