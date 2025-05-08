"""
This module provides the `BaseMDCalculator` class, which is used to perform
Molecular Dynamics (MD) simulations.

The `BaseMDCalculator` class allows users to set up and run MD simulations with different ensembles,
including NVE, NVT (Nose-Hoover), and NPT (Nose-Hoover). The calculator is designed to handle advanced
MD settings such as velocity initialization, pressure control, and symmetry constraints.
"""
from __future__ import annotations

from abc import ABC
from typing import Literal, TYPE_CHECKING

import numpy as np
from ase import units
from ase.calculators.calculator import Calculator
from ase.constraints import FixSymmetry
from ase.md import MDLogger, VelocityVerlet
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen, Inhomogeneous_NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from materialsframework.tools.trajectory import TrajectoryObserver

if TYPE_CHECKING:
    from ase import Atoms
    from typing import Any

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class BaseMDCalculator(ABC):
    """
    A calculator class for performing Molecular Dynamics (MD) simulations using universal potentials.

    The `BaseMDCalculator` class supports different ensembles such as NVE, NVT (Nose-Hoover), and NPT (Nose-Hoover),
    and provides a range of customizable parameters to control the simulation environment, including temperature,
    pressure, and timestep. This class also integrates constraints like fixing symmetry and initializing velocities
    to prepare the system for MD simulations.
    """

    def __init__(
            self,
            fix_symmetry: bool = False,
            ensemble: Literal["nve", "nvt_nose_hoover", "npt_nose_hoover"] = "nve",
            timestep: float = 1.0,  # fs
            temperature: int = 300,  # K
            pressure: float = 1,  # atm
            ttime: float = 10.0,  # fs
            pfactor: float = 75.0 ** 2.0,  # fs ** 2
            taut: float = 0.5e3,  # fs
            taup: float = 1e3,  # fs
            compressibility: float = 5e-7,  # 1/bar
            stationary: bool = True,
            zero_rotation: bool = True,
            logfile: str | None = None,
            loginterval: int = 1,
            interval: int = 1,
    ) -> None:
        """
        Initializes the `BaseMDCalculator` with the specified parameters for running MD simulations.

        Args:
            fix_symmetry (bool, optional): Whether to apply symmetry constraints during the simulation. Defaults to False.
            ensemble (Literal["nve", "nvt_nose_hoover", "npt_nose_hoover"], optional): The ensemble to use in the simulation. Defaults to "nve".
            timestep (float, optional): The timestep for the MD simulation in femtoseconds (fs). Defaults to 1.0 fs.
            temperature (int, optional): The temperature in Kelvin (K) for the MD simulation. Defaults to 300 K.
            pressure (float, optional): The pressure in atmospheres (atm) for the NPT ensemble. Defaults to 1 atm.
            ttime (float, optional): The time constant for temperature control in femtoseconds (fs). Defaults to 10.0 fs.
            pfactor (float, optional): Pressure factor for the NPT ensemble in fs^2. Defaults to 75.0^2 fs^2.
            taut (float, optional): Time constant for Berendsen temperature coupling in fs. Defaults to 0.5e3 fs.
            taup (float, optional): Time constant for Berendsen pressure coupling in fs. Defaults to 1e3 fs.
            compressibility (float, optional): Compressibility for the NPT ensemble in 1/bar. Defaults to 5e-7 1/bar.
            stationary (bool, optional): Whether to set the center-of-mass motion to zero. Defaults to True.
            zero_rotation (bool, optional): Whether to set the total angular momentum to zero. Defaults to True.
            logfile (str | None, optional): The file to log simulation output. If None, no logging occurs. Defaults to None.
            loginterval (int, optional): The interval at which to log the simulation results. Defaults to 1 (every step).
            interval (int, optional): The interval at which to record the simulation trajectory. Defaults to 1 (every step).

        Raises:
            ValueError: If an unsupported ensemble type is provided.
        """
        if ensemble not in ["nve", "nvt_nose_hoover", "npt_nose_hoover", "npt_berendsen", "inhomogeneous_npt_berendsen"]:
            raise ValueError("Ensemble must be one of 'nve', 'nvt_nose_hoover', 'npt_nose_hoover'")

        self.fix_symmetry: bool = fix_symmetry
        self.ensemble: str = ensemble
        self.timestep: float = timestep
        self.temperature: float = temperature
        self.pressure: float = pressure
        self.pfactor: float = pfactor
        self.taut: float = taut
        self.taup: float = taup
        self.compressibility: float = compressibility
        self.ttime: float = ttime
        self.stationary: bool = stationary
        self.zero_rotation: bool = zero_rotation
        self.logfile: str | None = logfile
        self.loginterval: int = loginterval
        self.interval: int = interval
        self.ase_adaptor = AseAtomsAdaptor()

        self.dyn = None
        self.atoms = None
        self.trajectory = None
        self.results = None

    @property
    def calculator(self) -> Calculator:
        """
        Returns the ASE Calculator object associated with this instance.

        This property must be implemented in subclasses of BaseMDCalculator.
        The returned Calculator object is used to perform the molecular dynamics
        calculation of structures within the run method.

        Raises:
            NotImplementedError: If the subclass does not implement this property.

        Returns:
            Calculator: An ASE Calculator instance configured for the specific
            molecular dynamics task.
        """
        raise NotImplementedError("Subclasses must implement the 'calculator' property to return a valid ASE Calculator instance.")

    def _initialize_npt_nose_hoover(self, ase_atoms: Atoms) -> None:
        """
        Initializes the NPT Nose-Hoover ensemble for MD simulations.

        Args:
            ase_atoms (Atoms): The ASE atoms object used in the simulation.
        """
        self._upper_triangular_cell(ase_atoms)
        self.dyn = NPT(
                atoms=ase_atoms,
                timestep=self.timestep * units.fs,
                temperature_K=self.temperature,
                externalstress=self.pressure * 1.01325 * units.bar,
                ttime=self.ttime * units.fs,
                pfactor=self.pfactor * units.fs,
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
                timestep=self.timestep * units.fs,
                temperature_K=self.temperature,
                ttime=self.ttime * units.fs,
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
                timestep=self.timestep * units.fs,
        )

    def _initialize_npt_berendsen(self, ase_atoms: Atoms) -> None:
        """
        Initializes the NPT Berendsen ensemble for MD simulations.

        Args:
            ase_atoms (Atoms): The ASE atoms object used in the simulation.
        """
        self.dyn = NPTBerendsen(
                atoms=ase_atoms,
                timestep=self.timestep * units.fs,
                temperature=self.temperature,
                pressure_au=self.pressure * 1.01325 * units.bar,
                taut=self.taut * units.fs,
                taup=self.taup * units.fs,
                compressibility=self.compressibility / units.bar,
        )

    def _initialize_inhomogeneous_npt_berendsen(self, ase_atoms: Atoms) -> None:
        """
        Initializes the Inhomogeneous NPT Berendsen ensemble for MD simulations.

        Args:
            ase_atoms (Atoms): The ASE atoms object used in the simulation.
        """
        self.dyn = Inhomogeneous_NPTBerendsen(
                atoms=ase_atoms,
                timestep=self.timestep * units.fs,
                temperature=self.temperature,
                pressure_au=self.pressure * 1.01325 * units.bar,
                taut=self.taut * units.fs,
                taup=self.taup * units.fs,
                compressibility=self.compressibility / units.bar,
        )

    def run(
            self,
            structure: Atoms | Structure | Molecule,
            steps: int
    ) -> dict[str, Any]:
        """
        Executes the Molecular Dynamics (MD) simulation using the specified calculator.

        This method performs the simulation based on the provided structure and
        simulation parameters such as the ensemble type and the number of MD steps.

        Args:
            structure (Atoms | Structure | Molecule): The input atomic structure for the MD simulation.
            steps (int): The number of MD steps to perform.

        Returns:
            dict[str, list]: A dictionary containing the results of the MD simulation, including
                             total energy, potential energy, kinetic energy, forces, stresses, and temperature.
        """

        ase_atoms = self.ase_adaptor.get_atoms(structure) \
            if isinstance(structure, (Structure, Molecule)) \
            else structure.copy()

        MaxwellBoltzmannDistribution(ase_atoms, temperature_K=self.temperature)

        if self.stationary:
            Stationary(ase_atoms)
        if self.zero_rotation:
            ZeroRotation(ase_atoms)

        ase_atoms.calc = self.calculator

        if self.fix_symmetry:
            ase_atoms.set_constraint(FixSymmetry(ase_atoms))

        if self.ensemble.lower() == "npt_nose_hoover":
            self._initialize_npt_nose_hoover(ase_atoms)
        elif self.ensemble.lower() == "nvt_nose_hoover":
            self._initialize_nvt_nose_hoover(ase_atoms)
        elif self.ensemble.lower() == "nve":
            self._initialize_nve(ase_atoms)
        elif self.ensemble.lower() == "npt_berendsen":
            self._initialize_npt_berendsen(ase_atoms)
        elif self.ensemble.lower() == "inhomogeneous_npt_berendsen":
            self._initialize_inhomogeneous_npt_berendsen(ase_atoms)

        if self.logfile:
            self._initialize_logger(ase_atoms)

        self.trajectory = TrajectoryObserver(
                ase_atoms,
                include_temperature=True
        )
        self.dyn.attach(self.trajectory, interval=self.interval)

        self.dyn.run(steps)

        self.results = {
                "total_energy": self.trajectory.total_energies,
                "potential_energy": self.trajectory.potential_energies,
                "kinetic_energy": self.trajectory.kinetic_energies,
                "forces": self.trajectory.forces,
                "stresses": self.trajectory.stresses,
                "temperature": self.trajectory.temperatures,
                "final_structure": self.ase_adaptor.get_structure(self.dyn.atoms),
        }

        return self.results

    def _initialize_logger(self, ase_atoms) -> None:
        """
        Initializes the logger for the MD simulation.

        Args:
            ase_atoms (Atoms): The ASE atoms object used in the simulation.
        """
        logger = MDLogger(
                dyn=self.dyn,
                atoms=ase_atoms,
                logfile=self.logfile,
                stress=True,
        )
        self.dyn.attach(logger, interval=self.loginterval)

    @staticmethod
    def _upper_triangular_cell(atoms: Atoms) -> None:
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
