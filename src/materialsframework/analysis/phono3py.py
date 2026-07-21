"""This module provides a class to calculate phonon properties of a structure using Phono3py.

The `Phono3pyAnalyzer` class generates displaced structures, computes forces with the provided
calculator, and calculates thermal conductivity using the RTA or LBTE methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from materialsframework.analysis.base import BaseAnalyzer
from materialsframework.analysis.utils import require_properties
from materialsframework.transformations.phono3py import (
    Phono3pyDisplacementTransformation,
)
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from typing import Literal

    from ase import Atoms
    from numpy.typing import ArrayLike
    from phono3py.conductivity.direct_solution import ConductivityLBTE
    from phono3py.conductivity.rta import ConductivityRTA
    from pymatgen.core import Structure

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class Phono3pyAnalyzer(BaseAnalyzer):
    """A class used to calculate phonon properties using Phono3py.

    Generates displaced supercells, calculates forces with the provided calculator, and computes thermal
    conductivity using the Relaxation Time Approximation (RTA) or the Linearized Boltzmann Transport
    Equation (LBTE) method.
    """

    def __init__(
        self,
        calculator: BaseCalculator | None = None,
        phono3py_transformation: Phono3pyDisplacementTransformation | None = None,
    ) -> None:
        """Initializes the `Phono3pyAnalyzer` object.

        Args:
            calculator (BaseCalculator, optional): The calculator used to compute forces and energies.
            phono3py_transformation (Phono3pyDisplacementTransformation, optional): The transformation object used to
                generate displaced structures.
        """
        super().__init__(calculator)
        self._phono3py_transformation = phono3py_transformation

    @require_properties("forces")
    def calculate(
        self,
        structure: Structure | Atoms,
        is_relaxed: bool = False,
        distance: float = 0.01,
        supercell_matrix: list | None = None,
        primitive_matrix: list | str | None = None,
        phonon_supercell_matrix: list | None = None,
        mesh: ArrayLike | float | None = None,
        is_lbte: bool = False,
        is_isotope: bool = False,
        conductivity_type: Literal["wigner", "kubo"] | None = None,
        boundary_mfp: float | None = None,
        t_min: float | None = 0,
        t_max: float | None = 1000,
        t_step: float | None = 10,
        log_level: Literal[0, 1, 2] = 0,
    ) -> dict[str, ConductivityRTA | ConductivityLBTE]:
        """Calculates the phonon properties of the given structure, including thermal conductivity.

        This method generates displaced supercells using Phono3py, calculates the forces using the provided calculator,
        and computes thermal conductivity based on the chosen method (RTA or LBTE).

        Args:
            structure (Structure | Atoms): The structure to calculate phonon properties for.
            is_relaxed (bool, optional): Whether the input structure is already relaxed. Defaults to False.
            distance (float, optional): The distance to displace atoms for force calculations. Defaults to 0.01.
            supercell_matrix (list, optional): The supercell matrix for generating supercells. Defaults to None.
            primitive_matrix (list | str, optional): The primitive matrix for generating supercells. Defaults to None.
            phonon_supercell_matrix (list, optional): The supercell matrix for phonon calculations. Defaults to None.
            mesh (ArrayLike | float, optional): The mesh numbers for phonon calculations. Defaults to [20, 20, 20].
            is_lbte (bool, optional): Whether to use the Linearized Boltzmann Transport Equation (LBTE). Defaults to False.
            is_isotope (bool, optional): Whether to include isotope scattering in the calculations. Defaults to False.
            conductivity_type (Literal["wigner", "kubo"], optional): The type of conductivity calculation to perform. Defaults to None.
            boundary_mfp (float, optional): Mean free path in micrometre to calculate simple boundary scattering contribution to thermal conductivity. None ignores this contribution.
            t_min (float, optional): The minimum temperature for thermal conductivity calculations. Defaults to 0.
            t_max (float, optional): The maximum temperature for thermal conductivity calculations. Defaults to 1000.
            t_step (float, optional): The step size for temperature increments. Defaults to 10.
            log_level (Literal[0, 1, 2], optional): The log level for Phono3py. Defaults to 0.

        Returns:
            dict[str, ConductivityRTA | ConductivityLBTE]: Dictionary with key:
                - ``thermal_conductivity``: Thermal conductivity object (RTA or LBTE).

        Raises:
            ValueError: If the calculator object does not have the 'forces' property implemented.
        """
        structure = self._ensure_relaxed(structure, is_relaxed)

        mesh = mesh or [20, 20, 20]

        phono3py_result = self.phono3py_transformation.apply_transformation(
            structure=structure,
            distance=distance,
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            phonon_supercell_matrix=phonon_supercell_matrix,
            log_level=log_level,
        )

        phonon = phono3py_result["phonon"]
        supercells_with_displacements = phono3py_result["supercells_with_displacements"]
        phonon_supercells_with_displacements = phono3py_result["phonon_supercells_with_displacements"]
        self._produce_force_constants(phonon, supercells_with_displacements, phonon_supercells_with_displacements)

        phonon.mesh_numbers = mesh
        phonon.init_phph_interaction()
        phonon.run_phonon_solver()

        # Thermal Conductivity
        phonon.run_thermal_conductivity(
            is_LBTE=is_lbte,
            is_isotope=is_isotope,
            conductivity_type=conductivity_type,
            boundary_mfp=boundary_mfp,
            temperatures=np.arange(t_min, t_max + t_step, t_step),
        )
        thermal_conductivity: ConductivityRTA | ConductivityLBTE = phonon.thermal_conductivity

        return {"thermal_conductivity": thermal_conductivity}

    @lazy_property("_phono3py_transformation")
    def phono3py_transformation(self) -> Phono3pyDisplacementTransformation:
        """Returns the Phono3py transformation object used to generate displaced structures.

        Returns:
            Phono3pyDisplacementTransformation: The transformation object used for phonon property calculations.
        """
        return Phono3pyDisplacementTransformation()

    def _produce_force_constants(self, phonon, supercells_with_displacements, phonon_supercells_with_displacements) -> None:
        """Produces the force constants using the forces calculated from the calculator.

        This method calculates the forces on the displaced atoms using the provided calculator and then
        generates the second- and third-order force constants required for phonon calculations.

        Args:
            phonon (Phono3py): The `Phono3py` object to produce force constants for.
            supercells_with_displacements (list[Structure]): Displaced supercells for third-order force constants.
            phonon_supercells_with_displacements (list[Structure]): Displaced supercells for phonon (second-order)
                force constants.
        """
        forces = [
            self.calculator.calculate(displaced_structure)["forces"] for displaced_structure in supercells_with_displacements
        ]
        phonon.forces = np.array(forces)

        phonon_forces = [
            self.calculator.calculate(displaced_structure)["forces"]
            for displaced_structure in phonon_supercells_with_displacements
        ]
        phonon.phonon_forces = np.array(phonon_forces)

        phonon.produce_fc3()
        phonon.produce_fc2()
