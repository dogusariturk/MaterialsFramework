"""This module provides the `BaseMDCalculator` class for Molecular Dynamics simulations.

`BaseMDCalculator` sets up and runs MD simulations with different ensembles, including NVE,
NVT (Nose-Hoover), NPT (Nose-Hoover), NPT (Berendsen), and Inhomogeneous NPT (Berendsen). It
handles advanced MD settings such as velocity initialization, pressure control, and symmetry
constraints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import numpy as np
from ase import units
from ase.md import MDLogger, VelocityVerlet
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.velocitydistribution import (
    Stationary,
    ZeroRotation,
    thermalize_momenta,
)
from pymatgen.io.ase import AseAtomsAdaptor

from materialsframework.tools.trajectory import TrajectoryObserver
from materialsframework.utils import to_atoms

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Molecule, Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class BaseMDCalculator(ABC):
    """A calculator class for performing Molecular Dynamics (MD) simulations using universal potentials.

    Supports NVE, NVT (Nose-Hoover), and NPT (Nose-Hoover) ensembles, with customizable
    parameters for temperature, pressure, and timestep. Also applies constraints such as
    fixing symmetry and initializing velocities before a simulation starts.
    """

    def __init__(
        self,
        ensemble: Literal[
            "nve",
            "nvt_nose_hoover",
            "npt_nose_hoover",
            "npt_berendsen",
            "inhomogeneous_npt_berendsen",
        ] = "nve",
        timestep: float = 1.0,  # fs
        temperature: int = 300,  # K
        pressure: float = 1,  # atm
        ttime: float = 10.0,  # fs
        pfactor: float = 75.0**2.0,  # fs ** 2
        taut: float = 0.5e3,  # fs
        taup: float = 1e3,  # fs
        compressibility: float = 5e-7,  # 1/bar
        mask: tuple[int] = (1, 1, 1),
        stationary: bool = True,
        zero_rotation: bool = True,
        logfile: str | None = None,
        loginterval: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initializes the `BaseMDCalculator` with the specified parameters for running MD simulations.

        Args:
            ensemble (Literal["nve", "nvt_nose_hoover", "npt_nose_hoover"], optional): The ensemble to use in the
                simulation. Defaults to "nve".
            timestep (float, optional): The timestep for the MD simulation in femtoseconds (fs). Defaults to 1.0 fs.
            temperature (int, optional): The temperature in Kelvin (K) for the MD simulation. Defaults to 300 K.
            pressure (float, optional): The pressure in atmospheres (atm) for the NPT ensemble. Defaults to 1 atm.
            ttime (float, optional): The time constant for temperature control in femtoseconds (fs). Defaults to 10.0 fs.
            pfactor (float, optional): Pressure factor for the NPT ensemble in fs^2. Defaults to 75.0^2 fs^2.
            taut (float, optional): Time constant for Berendsen temperature coupling in fs. Defaults to 0.5e3 fs.
            taup (float, optional): Time constant for Berendsen pressure coupling in fs. Defaults to 1e3 fs.
            compressibility (float, optional): Compressibility for the NPT ensemble in 1/bar. Defaults to 5e-7 1/bar.
            mask (tuple[int], optional): Specifies which axes participate in the barostat for the Inhomogeneous NPT
                Berendsen ensemble. Defaults to (1, 1, 1).
            stationary (bool, optional): Whether to set the center-of-mass motion to zero. Defaults to True.
            zero_rotation (bool, optional): Whether to set the total angular momentum to zero. Defaults to True.
            logfile (str | None, optional): The file to log simulation output. If None, no logging occurs. Defaults to None.
            loginterval (int, optional): The interval at which to log the simulation results. Defaults to 1 (every step).
            **kwargs: Forwarded to the next class in the MRO, so cooperative subclasses can chain a
                single `super().__init__(**kwargs)` call instead of splitting kwargs by hand.

        Raises:
            ValueError: If an unsupported ensemble type is provided.
        """
        if ensemble not in [
            "nve",
            "nvt_nose_hoover",
            "npt_nose_hoover",
            "npt_berendsen",
            "inhomogeneous_npt_berendsen",
        ]:
            raise ValueError(
                "Ensemble must be one of 'nve', 'nvt_nose_hoover', 'npt_nose_hoover', 'npt_berendsen', or 'inhomogeneous_npt_berendsen'."
            )

        self.ensemble: str = ensemble
        self.timestep: float = timestep
        self.temperature: float = temperature
        self.pressure: float = pressure
        self.pfactor: float = pfactor
        self.taut: float = taut
        self.taup: float = taup
        self.compressibility: float = compressibility
        self.mask: tuple[int] = mask
        self.ttime: float = ttime
        self.stationary: bool = stationary
        self.zero_rotation: bool = zero_rotation
        self.logfile: str | None = logfile
        self.loginterval: int = loginterval
        self.ase_adaptor = AseAtomsAdaptor()

        super().__init__(**kwargs)

    @property
    @abstractmethod
    def calculator(self) -> Calculator:
        """Returns the ASE Calculator object associated with this instance.

        Subclasses of BaseMDCalculator must implement this property; the returned Calculator
        object performs the molecular dynamics calculation of structures within `run()`.

        Raises:
            NotImplementedError: If the subclass does not implement this property.

        Returns:
            Calculator: An ASE Calculator instance configured for the specific
            molecular dynamics task.
        """
        raise NotImplementedError("Subclasses must implement the 'calculator' property to return a valid ASE Calculator instance.")

    def _initialize_npt_nose_hoover(self, ase_atoms: Atoms) -> NPT:
        """Initializes the NPT Nose-Hoover ensemble for MD simulations.

        Args:
            ase_atoms (Atoms): The ASE atoms object used in the simulation.

        Returns:
            NPT: The initialized NPT dynamics object.
        """
        self._upper_triangular_cell(ase_atoms)
        return NPT(
            atoms=ase_atoms,
            timestep=self.timestep * units.fs,
            temperature_K=self.temperature,
            externalstress=self.pressure * 1.01325 * units.bar,
            ttime=self.ttime * units.fs,
            pfactor=self.pfactor * units.fs,
        )

    def _initialize_nvt_nose_hoover(self, ase_atoms: Atoms) -> NPT:
        """Initializes the NVT Nose-Hoover ensemble for MD simulations.

        Args:
            ase_atoms (Atoms): The ASE atoms object used in the simulation.

        Returns:
            NPT: The initialized NPT dynamics object.
        """
        self._upper_triangular_cell(ase_atoms)
        return NPT(
            atoms=ase_atoms,
            timestep=self.timestep * units.fs,
            temperature_K=self.temperature,
            ttime=self.ttime * units.fs,
            pfactor=None,
        )

    def _initialize_nve(self, ase_atoms: Atoms) -> VelocityVerlet:
        """Initializes the NVE ensemble for MD simulations.

        Args:
            ase_atoms (Atoms): The ASE atoms object used in the simulation.

        Returns:
            VelocityVerlet: The initialized VelocityVerlet dynamics object.
        """
        return VelocityVerlet(
            atoms=ase_atoms,
            timestep=self.timestep * units.fs,
        )

    def _initialize_npt_berendsen(self, ase_atoms: Atoms) -> NPTBerendsen:
        """Initializes the NPT Berendsen ensemble for MD simulations.

        Args:
            ase_atoms (Atoms): The ASE atoms object used in the simulation.

        Returns:
            NPTBerendsen: The initialized NPTBerendsen dynamics object.
        """
        return NPTBerendsen(
            atoms=ase_atoms,
            timestep=self.timestep * units.fs,
            temperature=self.temperature,
            pressure_au=self.pressure * 1.01325 * units.bar,
            taut=self.taut * units.fs,
            taup=self.taup * units.fs,
            compressibility_au=self.compressibility / units.bar,
        )

    def _initialize_inhomogeneous_npt_berendsen(self, ase_atoms: Atoms) -> Inhomogeneous_NPTBerendsen:
        """Initializes the Inhomogeneous NPT Berendsen ensemble for MD simulations.

        Args:
            ase_atoms (Atoms): The ASE atoms object used in the simulation.

        Returns:
            Inhomogeneous_NPTBerendsen: The initialized Inhomogeneous_NPTBerendsen dynamics object.
        """
        return Inhomogeneous_NPTBerendsen(
            atoms=ase_atoms,
            timestep=self.timestep * units.fs,
            temperature=self.temperature,
            pressure_au=self.pressure * 1.01325 * units.bar,
            taut=self.taut * units.fs,
            taup=self.taup * units.fs,
            compressibility_au=self.compressibility / units.bar,
            mask=self.mask,
        )

    def run(self, structure: Atoms | Structure | Molecule, steps: int) -> dict[str, Any]:
        """Executes the Molecular Dynamics (MD) simulation using the specified calculator.

        Args:
            structure (Atoms | Structure | Molecule): The input atomic structure for the MD simulation.
            steps (int): The number of MD steps to perform.

        Returns:
            dict[str, Any]: Dictionary with keys:
                - ``total_energy``: Total energies at each recorded MD step (eV).
                - ``potential_energy``: Potential energies at each recorded MD step (eV).
                - ``kinetic_energy``: Kinetic energies at each recorded MD step (eV).
                - ``forces``: Force arrays at each recorded MD step (eV/Å).
                - ``stresses``: Stress tensors at each recorded MD step.
                - ``temperature``: Temperatures at each recorded MD step (K).
                - ``velocities``: Velocity arrays at each recorded MD step.
                - ``final_structure``: Final structure as a pymatgen ``Structure``.
        """
        ase_atoms = to_atoms(structure)

        thermalize_momenta(ase_atoms, temperature_K=self.temperature)

        if self.stationary:
            Stationary(ase_atoms)
        if self.zero_rotation:
            ZeroRotation(ase_atoms)

        ase_atoms.calc = self.calculator

        if self.ensemble.lower() == "npt_nose_hoover":
            dyn = self._initialize_npt_nose_hoover(ase_atoms)
        elif self.ensemble.lower() == "nvt_nose_hoover":
            dyn = self._initialize_nvt_nose_hoover(ase_atoms)
        elif self.ensemble.lower() == "nve":
            dyn = self._initialize_nve(ase_atoms)
        elif self.ensemble.lower() == "npt_berendsen":
            dyn = self._initialize_npt_berendsen(ase_atoms)
        elif self.ensemble.lower() == "inhomogeneous_npt_berendsen":
            dyn = self._initialize_inhomogeneous_npt_berendsen(ase_atoms)

        if self.logfile:
            self._initialize_logger(dyn, ase_atoms)

        trajectory = TrajectoryObserver(ase_atoms, include_temperature=True, include_velocities=True)
        dyn.attach(trajectory, interval=self.interval)

        dyn.run(steps)

        return {
            "total_energy": trajectory.total_energies,
            "potential_energy": trajectory.potential_energies,
            "kinetic_energy": trajectory.kinetic_energies,
            "forces": trajectory.forces,
            "stresses": trajectory.stresses,
            "temperature": trajectory.temperatures,
            "velocities": trajectory.velocities,
            "final_structure": self.ase_adaptor.get_structure(dyn.atoms),
        }

    def _initialize_logger(self, dyn, ase_atoms) -> None:
        """Initializes the logger for the MD simulation.

        Args:
            dyn: The MD dynamics object being logged.
            ase_atoms (Atoms): The ASE atoms object used in the simulation.
        """
        logger = MDLogger(
            dyn=dyn,
            atoms=ase_atoms,
            logfile=self.logfile,
            stress=True,
        )
        dyn.attach(logger, interval=self.loginterval)

    @staticmethod
    def _upper_triangular_cell(atoms: Atoms) -> None:
        """Converts the unit cell of the provided atoms object to upper triangular form, as required for MD simulations.

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
            sin_p = (1 - cos_p**2) ** 0.5

            new_basis = [
                (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
                (0, b * sin_a, b * cos_a),
                (0, 0, c),
            ]

            atoms.set_cell(new_basis, scale_atoms=True)
