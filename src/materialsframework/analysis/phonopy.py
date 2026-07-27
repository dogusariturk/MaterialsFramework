"""This module provides a class to calculate phonon properties of a structure using Phonopy.

The `PhonopyAnalyzer` class computes the total density of states (DOS), projected DOS (PDOS), and
thermal properties from displaced structures and force constants generated with Phonopy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from materialsframework.analysis.base import BaseAnalyzer
from materialsframework.analysis.utils import require_properties
from materialsframework.transformations.phonopy import PhonopyDisplacementTransformation
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase import Atoms
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class PhonopyAnalyzer(BaseAnalyzer):
    """A class used to calculate phonon properties using Phonopy.

    Generates displaced supercells, calculates forces with a specified calculator, and computes the
    total density of states (DOS), projected DOS (PDOS), and thermal properties of a given structure.
    """

    def __init__(
        self,
        calculator: BaseCalculator | None = None,
        phonopy_transformation: PhonopyDisplacementTransformation | None = None,
    ) -> None:
        """Initializes the `PhonopyAnalyzer` object.

        Args:
            calculator (BaseCalculator, optional): The calculator used to compute forces and energies.
            phonopy_transformation (PhonopyDisplacementTransformation, optional): The transformation object used to
                generate displaced supercells.
        """
        super().__init__(calculator)
        self._phonopy_transformation = phonopy_transformation

    @require_properties("forces")
    def calculate(
        self,
        structure: Structure | Atoms,
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
        log_level: int = 0,
    ) -> dict[str, dict]:
        """Calculates the phonon properties of the given structure.

        This method generates displaced supercells using Phonopy, calculates the forces using the provided calculator,
        and computes the total density of states (DOS), projected DOS (PDOS), and thermal properties.

        Args:
            structure (Structure | Atoms): The structure to calculate phonon properties for.
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
            dict[str, dict]: Dictionary with keys:
                - ``total_dos``: Total phonon density of states payload.
                - ``thermal_properties``: Thermal-properties payload.
                - ``projected_dos``: Projected phonon density of states payload.

        Raises:
            ValueError: If the calculator object does not have the 'forces' property implemented.
        """
        structure = self._ensure_relaxed(structure, is_relaxed)

        mesh = mesh or [20, 20, 20]
        pdos_mesh = pdos_mesh or [10, 10, 10]

        phonopy_result = self.phonopy_transformation.apply_transformation(
            structure=structure,
            distance=distance,
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            log_level=log_level,
        )
        phonon = phonopy_result["phonon"]
        displaced_structures = phonopy_result["displaced_structures"]
        self._produce_force_constants(phonon, displaced_structures)

        phonon.run_mesh(mesh=mesh)

        # DOS
        phonon.run_total_dos(sigma=sigma, freq_min=freq_min, freq_max=freq_max, freq_pitch=freq_pitch)
        total_dos = {
            "frequency_points": phonon.total_dos.frequency_points,
            "total_dos": phonon.total_dos.dos,
        }

        # Thermal Properties
        phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)
        thermal_properties = {
            "temperatures": phonon.thermal_properties.temperatures,
            "free_energy": phonon.thermal_properties.free_energy,
            "entropy": phonon.thermal_properties.entropy,
            "heat_capacity": phonon.thermal_properties.heat_capacity,
        }

        # PDOS
        phonon.run_mesh(mesh=pdos_mesh, is_mesh_symmetry=False, with_eigenvectors=True)
        phonon.run_projected_dos()
        projected_dos = {
            "frequency_points": phonon.projected_dos.frequency_points,
            "projected_dos": phonon.projected_dos.projected_dos,
        }

        return {
            "total_dos": total_dos,
            "thermal_properties": thermal_properties,
            "projected_dos": projected_dos,
        }

    @lazy_property("_phonopy_transformation")
    def phonopy_transformation(self) -> PhonopyDisplacementTransformation:
        """Returns the Phonopy transformation object used to generate displaced structures.

        Returns:
            PhonopyDisplacementTransformation: The transformation object used for phonon property calculations.
        """
        return PhonopyDisplacementTransformation()

    def _produce_force_constants(self, phonon, displaced_structures) -> None:
        """Produces the force constants using the forces calculated from the calculator.

        This method calculates the forces on the displaced atoms using the provided calculator and generates
        the force constants required for phonon calculations.

        Args:
            phonon (Phonopy): The `Phonopy` object to produce force constants for.
            displaced_structures (list[Structure]): The displaced structures used to calculate forces.
        """
        forces = [self.calculator.calculate(displaced_structure)["forces"] for displaced_structure in displaced_structures]
        phonon.forces = forces
        phonon.produce_force_constants()
