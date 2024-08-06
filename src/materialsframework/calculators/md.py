"""
This module contains the M3GNetMDCalculator class, which is used to perform
Molecular Dynamics (MD) simulations using the M3GNet potential.
"""
from __future__ import annotations

import os
from typing import Literal, Optional, TYPE_CHECKING, Union

import matgl
import numpy as np
from ase import units
from ase.constraints import FixSymmetry
from ase.md import MDLogger, VelocityVerlet
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from matgl.ext.ase import PESCalculator

from materialsframework.tools.trajectory import TrajectoryObserver
from materialsframework.tools.typing import MDCalculator

if TYPE_CHECKING:
    from ase import Atoms
    from matgl.apps.pes import Potential
    from pymatgen.core import Structure

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class M3GNetMDCalculator(MDCalculator):
    """
    A class used to represent a M3GNet Molecular Dynamics (MD) Calculator.

    This class provides methods to perform Molecular Dynamics (MD) simulations using the M3GNet potential.
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
            logfile: Optional[str] = None,
            loginterval: int = 1,
    ) -> None:
        """
        Initializes the M3GNet Molecular Dynamics (MD) Calculator.

        Args:
            model (str): The M3GNet model to use. Defaults to "M3GNet-MP-2021.2.8-PES".
            ensemble (Literal["nve", "nvt_nose_hoover", "npt_nose_hoover"]): The ensemble to use. Defaults to "nve".
            timestep (float): The timestep in fs for the MD simulation. Defaults to 1.0 fs.
            temperature (int): The temperature in K for the MD simulation. Defaults to 300 K.
            pressure (float): The pressure in atm for the MD simulation. Defaults to 1 atm.
            ttime (float): The time constant for the thermostat. Defaults to 25.0 fs.
            pfactor (float): The pressure factor for the MD simulation. Defaults to 75.0 fs ** 2.0.
            stationary (bool): Whether to et the center-of-mass momentum to zero. Defaults to True.
            zero_rotation (bool): Whether to set the total angular momentum to zero. Defaults to True.
            logfile (Optional[str]): The logfile to save the results. Defaults to None.
            loginterval (int): The interval to log the results. Defaults to 1 (each step).
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
        self._logfile: Union[str, None] = logfile
        self._loginterval: int = loginterval

        self._potential = None

        self.dyn = None
        self.atoms = None
        self.trajectory = None
        self.results = None

    @property
    def potential(self) -> Potential:
        """
        Returns the M3GNet potential associated with this instance.

        If the potential has not been initialized yet, it will be loaded
        using the model attribute of this instance.

        Returns:
            Potential: The M3GNet potential associated with this instance.
        """
        if self._potential is None:
            self._potential = matgl.load_model(self._model)
        return self._potential

    def _initialize_npt_nose_hoover(self, ase_atoms: Atoms) -> None:
        """
        Initializes the NPT Nose-Hoover ensemble for the MD simulation.

        Args:
            ase_atoms (Atoms): The ASE atoms object.
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
        Initializes the NVT Nose-Hoover ensemble for the MD simulation.

        Args:
            ase_atoms (Atoms): The ASE atoms object.
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
        Initializes the NVE ensemble for the MD simulation.

        Args:
            ase_atoms (Atoms): The ASE atoms object.
        """
        self.dyn = VelocityVerlet(
                atoms=ase_atoms,
                timestep=self._timestep * units.fs,
        )

    def run(self, structure: Structure, steps: int) -> dict:
        """
        Performs the Molecular Dynamics (MD) simulation using the M3GNet potential.

        Args:
            structure (Structure): The input structure.
            steps (int): The number of MD steps.

        Returns:
            dict: A dictionary containing the results of the MD simulation.
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

        self.trajectory = TrajectoryObserver(ase_atoms)
        self.dyn.attach(self.trajectory, interval=1)

        self.dyn.run(steps)

        self.results = {
                "total_energy": self.trajectory.total_energies,
                "potential_energy": self.trajectory.potential_energies,
                "kinetic_energy": self.trajectory.kinetic_energies,
                "forces": self.trajectory.forces,
                "stresses": self.trajectory.stresses,
                "temperature": self.trajectory.temperature,
        }

        return self.results

    @staticmethod
    def _upper_triangular_cell(atoms) -> None:
        """
        Converts the cell of the atoms to upper triangular form.

        Args:
            atoms (Atoms): The ASE atoms object.

        NOTE: Adapted from the matgl code.
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

