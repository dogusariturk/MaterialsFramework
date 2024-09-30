"""
This module provides the `M3GNetMDCalculator` class, which is used to perform
Molecular Dynamics (MD) simulations using the M3GNet potential.

The `M3GNetMDCalculator` class allows users to set up and run MD simulations with different ensembles,
including NVE, NVT (Nose-Hoover), and NPT (Nose-Hoover). The calculator is designed to handle advanced
MD settings such as velocity initialization, pressure control, and symmetry constraints, all based on the
M3GNet potential model.
"""
from __future__ import annotations

import os
from typing import Literal, TYPE_CHECKING

import matgl
import numpy as np
from ase import units
from ase.constraints import FixSymmetry
from ase.md import MDLogger, VelocityVerlet
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from matgl.ext.ase import PESCalculator

from materialsframework.tools.trajectory import TrajectoryObserver

if TYPE_CHECKING:
    from ase import Atoms
    from matgl.apps.pes import Potential
    from pymatgen.core import Structure

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class M3GNetMDCalculator:
    """
    A calculator class for performing Molecular Dynamics (MD) simulations using the M3GNet potential.

    The `M3GNetMDCalculator` class supports different ensembles such as NVE, NVT (Nose-Hoover), and NPT (Nose-Hoover),
    and provides a range of customizable parameters to control the simulation environment, including temperature,
    pressure, and timestep. This class also integrates constraints like fixing symmetry and initializing velocities
    to prepare the system for MD simulations.
    """

    def __init__(
            self,
            model: str = "M3GNet-MP-2021.2.8-PES",
            fix_symmetry: bool = False,
            ensemble: Literal["nve", "nvt_nose_hoover", "npt_nose_hoover"] = "nve",
            timestep: float = 1.0,  # fs
            temperature: int = 300,  # K
            pressure: float = 1,  # atm
            ttime: float = 10.0,  # fs
            pfactor: float = 75.0 ** 2.0,  # fs ** 2
            stationary: bool = True,
            zero_rotation: bool = True,
            logfile: str | None = None,
            loginterval: int = 1,
    ) -> None:
        """
        Initializes the `M3GNetMDCalculator` with the specified parameters for running MD simulations.

        Args:
            model (str, optional): The M3GNet model to use for the MD simulation. Defaults to "M3GNet-MP-2021.2.8-PES".
            fix_symmetry (bool, optional): Whether to apply symmetry constraints during the simulation. Defaults to False.
            ensemble (Literal["nve", "nvt_nose_hoover", "npt_nose_hoover"], optional): The ensemble to use in the simulation. Defaults to "nve".
            timestep (float, optional): The timestep for the MD simulation in femtoseconds (fs). Defaults to 1.0 fs.
            temperature (int, optional): The temperature in Kelvin (K) for the MD simulation. Defaults to 300 K.
            pressure (float, optional): The pressure in atmospheres (atm) for the NPT ensemble. Defaults to 1 atm.
            ttime (float, optional): The time constant for temperature control in femtoseconds (fs). Defaults to 10.0 fs.
            pfactor (float, optional): Pressure factor for the NPT ensemble in fs^2. Defaults to 75.0^2 fs^2.
            stationary (bool, optional): Whether to set the center-of-mass motion to zero. Defaults to True.
            zero_rotation (bool, optional): Whether to set the total angular momentum to zero. Defaults to True.
            logfile (Optional[str], optional): The file to log simulation output. If None, no logging occurs. Defaults to None.
            loginterval (int, optional): The interval at which to log the simulation results. Defaults to 1 (every step).

        Raises:
            ValueError: If an unsupported ensemble type is provided.
        """
        if ensemble not in ["nve", "nvt_nose_hoover", "npt_nose_hoover"]:
            raise ValueError("Ensemble must be one of 'nve', 'nvt_nose_hoover', 'npt_nose_hoover'")

        self._model: str = model
        self._fix_symmetry: bool = fix_symmetry
        self._ensemble: str = ensemble
        self._timestep: float = timestep
        self._temperature: float = temperature
        self._pressure: float = pressure
        self._pfactor: float = pfactor
        self._ttime: float = ttime
        self._stationary: bool = stationary
        self._zero_rotation: bool = zero_rotation
        self._logfile: str | None = logfile
        self._loginterval: int = loginterval

        self._potential = None

        self.dyn = None
        self.atoms = None
        self.trajectory = None
        self.results = None

    @property
    def potential(self) -> Potential:
        """
        Loads and returns the M3GNet potential associated with this instance.

        If the potential has not already been initialized, this property will load
        it using the specified model. The loaded potential is then used for all
        subsequent MD simulations.

        Returns:
            Potential: The loaded M3GNet potential for the MD simulations.
        """
        if self._potential is None:
            self._potential = matgl.load_model(self._model)
        return self._potential

    def _initialize_npt_nose_hoover(self, ase_atoms: Atoms) -> None:
        """
        Initializes the NPT Nose-Hoover ensemble for MD simulations.

        Args:
            ase_atoms (Atoms): The ASE atoms object used in the simulation.
        """
        self._upper_triangular_cell(ase_atoms)
        self.dyn = NPT(
                atoms=ase_atoms,
                timestep=self._timestep * units.fs,
                temperature_K=self._temperature,
                externalstress=self._pressure * 1.01325 * units.bar,
                ttime=self._ttime * units.fs,
                pfactor=self._pfactor * units.fs,
        )

    def _initialize_nvt_nose_hoover(self, ase_atoms: Atoms) -> None:
        """
        Initializes the NVT Nose-Hoover ensemble for MD simulations.

        Args:
            ase_atoms (Atoms): The ASE atoms object used in the simulation.
        """
        self._upper_triangular_cell(ase_atoms)
        self.dyn = NPT(
                atoms=ase_atoms,
                timestep=self._timestep * units.fs,
                temperature_K=self._temperature,
                ttime=self._ttime * units.fs,
                pfactor=None,
        )

    def _initialize_nve(self, ase_atoms: Atoms) -> None:
        """
        Initializes the NVE ensemble for MD simulations.

        Args:
            ase_atoms (Atoms): The ASE atoms object used in the simulation.
        """
        self.dyn = VelocityVerlet(
                atoms=ase_atoms,
                timestep=self._timestep * units.fs,
        )

    def run(self, structure: Structure, steps: int) -> dict[str, list]:
        """
        Executes the Molecular Dynamics (MD) simulation using the M3GNet potential.

        This method performs the simulation based on the provided structure and
        simulation parameters such as the ensemble type and the number of MD steps.

        Args:
            structure (Structure): The input atomic structure for the MD simulation.
            steps (int): The number of MD steps to perform.

        Returns:
            dict: A dictionary containing the results of the MD simulation, including
                  total energy, potential energy, kinetic energy, forces, stresses, and temperature.
        """
        ase_atoms = structure.to_ase_atoms()

        MaxwellBoltzmannDistribution(ase_atoms, temperature_K=self._temperature)

        if self._stationary:
            Stationary(ase_atoms)
        if self._zero_rotation:
            ZeroRotation(ase_atoms)

        ase_atoms.calc = PESCalculator(potential=self.potential)

        if self._fix_symmetry:
            ase_atoms.set_constraint(FixSymmetry(ase_atoms))

        if self._ensemble.lower() == "npt_nose_hoover":
            self._initialize_npt_nose_hoover(ase_atoms)

        if self._ensemble.lower() == "nvt_nose_hoover":
            self._initialize_nvt_nose_hoover(ase_atoms)

        if self._ensemble.lower() == "nve":
            self._initialize_nve(ase_atoms)

        if self._logfile:
            logger = MDLogger(
                    dyn=self.dyn,
                    atoms=ase_atoms,
                    logfile=self._logfile,
                    header=True,
                    stress=True,
                    peratom=False,
            )
            self.dyn.attach(logger, interval=self._loginterval)

        self.trajectory = TrajectoryObserver(ase_atoms, include_temperature=True)
        self.dyn.attach(self.trajectory, interval=1)

        self.dyn.run(steps)

        self.results = {
                "total_energy": self.trajectory.total_energies,
                "potential_energy": self.trajectory.potential_energies,
                "kinetic_energy": self.trajectory.kinetic_energies,
                "forces": self.trajectory.forces,
                "stresses": self.trajectory.stresses,
                "temperature": self.trajectory.temperatures,
        }

        return self.results

    @staticmethod
    def _upper_triangular_cell(atoms) -> None:
        """
        Converts the unit cell of the provided atoms object to upper triangular form.

        This operation ensures that the cell parameters are in a suitable form for
        MD simulations.

        Args:
            atoms (Atoms): The ASE atoms object whose cell will be converted.

        Note:
            This method is adapted from the matgl code.
        """
        if not NPT._isuppertriangular(atoms.get_cell()):
            a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
            angles = np.radians((alpha, beta, gamma))
            sin_a, sin_b, _sin_g = np.sin(angles)
            cos_a, cos_b, cos_g = np.cos(angles)
            cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
            cos_p = np.clip(cos_p, -1, 1)
            sin_p = (1 - cos_p ** 2) ** 0.5

            new_basis = [
                    (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
                    (0, b * sin_a, b * cos_a),
                    (0, 0, c),
            ]

            atoms.set_cell(new_basis, scale_atoms=True)

