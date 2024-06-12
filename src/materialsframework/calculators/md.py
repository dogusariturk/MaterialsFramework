"""
This module contains the M3GNetMDCalculator class, which is used to perform
Molecular Dynamics (MD) simulations using the M3GNet potential.
"""
from __future__ import annotations

import collections
import os
from typing import Literal, Optional, TYPE_CHECKING, Union

import matgl
import numpy as np
import pandas as pd
from ase import units
from ase.md import VelocityVerlet
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from matgl.ext.ase import PESCalculator

from materialsframework.calculators.typing import Calculator

if TYPE_CHECKING:
    from ase import Atoms
    from matgl.apps.pes import Potential
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class M3GNetMDCalculator(Calculator):
    """
    A class used to represent a M3GNet Molecular Dynamics (MD) Calculator.

    This class provides methods to perform Molecular Dynamics (MD) simulations using the M3GNet potential.
    """

    def __init__(
            self,
            steps: int,
            avg_start: float,
            avg_end: float,
            model: str = "M3GNet-MP-2021.2.8-PES",
            ensemble: Literal["nve", "nvt_nose_hoover", "npt_nose_hoover"] = "nvt_nose_hoover",
            timestep: float = 1.0,
            temperature: int = 300,
            pressure: float = 1.01325 * units.bar,
            pfactor: float = 75.0 ** 2.0,
            external_stress: Union[float, ArrayLike] = 0.0,
            ttime: float = 25.0,
            logfile: Optional[str] = None,
            loginterval: int = 1,
            append_trajectory: bool = False,
            mask: Union[tuple, ArrayLike] = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ) -> None:
        """
        Initializes the M3GNet Molecular Dynamics (MD) Calculator.


        Args:
            steps (int): The number of MD steps.
            avg_start (float): The starting step in ps for the average.
            avg_end (float): The ending step in ps for the average.
            model (str): The M3GNet model to use. Defaults to "M3GNet-MP-2021.2.8-PES".
            ensemble (Literal["nvt_nose_hoover"]): The ensemble to use. Defaults to "nvt_nose_hoover".
            timestep (float): The timestep for the MD simulation. Defaults to 1.0 fs.
            temperature (int): The temperature for the MD simulation. Defaults to 300 K.
            pressure (float): The pressure for the MD simulation. Defaults to 1.01325 * units.bar.
            pfactor (float): The pressure factor for the MD simulation. Defaults to 75.0 fs ** 2.0.
            external_stress (Union[float, ArrayLike]): The external stress for the MD simulation. Defaults to 0.0.
            ttime (float): The time constant for the thermostat. Defaults to 25.0 fs.
            logfile (Optional[str]): The logfile to save the results. Defaults to None.
            loginterval (int): The interval to log the results. Defaults to 1.
            append_trajectory (bool): Whether to append the trajectory. Defaults to False.
            mask (Union[tuple, ArrayLike]): The mask for the MD simulation. Defaults to ((1, 0, 0), (0, 1, 0), (0, 0, 1)).
        """
        if avg_end <= avg_start:
            raise ValueError("avg_end must be greater than avg_start")

        if ensemble not in ["nve", "nvt_nose_hoover", "npt_nose_hoover"]:
            raise ValueError("Ensemble must be one of 'nve', 'nvt_nose_hoover', 'npt_nose_hoover'")

        self._steps: int = steps
        self._avg_start: int = avg_start
        self._avg_end: int = avg_end
        self._model: str = model
        self._ensemble: str = ensemble
        self._timestep: float = timestep
        self._temperature: float = temperature
        self._pressure: float = pressure
        self._pfactor: float = pfactor
        self._external_stress: Union[float, ArrayLike] = external_stress
        self._ttime: float = ttime
        self._logfile: Union[str, None] = logfile
        self._loginterval: int = loginterval
        self._append_trajectory: bool = append_trajectory
        self._mask: Union[tuple, ArrayLike, None] = mask

        self._potential = None

        self.dyn = None
        self.atoms = None
        self.trajectory = None

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

    def _get_average_results(self) -> dict:
        """
        Returns the average results of the trajectory.

        Returns:
            dict: A dictionary containing the average results of the trajectory.
        """
        avg_start_steps = round(self._avg_start * 1e3 / self._timestep)
        avg_end_steps = round(self._avg_end * 1e3 / self._timestep)

        energy = np.array(self.trajectory.energies[avg_start_steps:avg_end_steps])
        forces = np.array(self.trajectory.forces[avg_start_steps:avg_end_steps])
        stresses = np.array(self.trajectory.stresses[avg_start_steps:avg_end_steps])
        temperature = np.array(self.trajectory.temperature[avg_start_steps:avg_end_steps])

        return {
                "average_energy": np.mean(energy),
                "average_forces": np.mean(forces, axis=0),
                "average_stresses": np.mean(stresses, axis=0),
                "average_temperature": np.mean(temperature),
        }

    def _initialize_npt_nose_hoover(self, ase_atoms) -> None:
        """
        Initializes the NPT Nose-Hoover ensemble for the MD simulation.

        Args:
            ase_atoms (Atoms): The ASE atoms object.
        """
        self.dyn = NPT(
                atoms=ase_atoms,
                timestep=self._timestep * units.fs,
                temperature_K=self._temperature,
                externalstress=self._external_stress,
                ttime=self._ttime * units.fs,
                pfactor=self._pfactor * units.fs,
                mask=np.array(self._mask),
                trajectory=self.trajectory,
                logfile=self._logfile,
                loginterval=self._loginterval,
                append_trajectory=self._append_trajectory,
        )

    def _initialize_nvt_nose_hoover(self, ase_atoms) -> None:
        """
        Initializes the NVT Nose-Hoover ensemble for the MD simulation.

        Args:
            ase_atoms (Atoms): The ASE atoms object.
        """
        self.dyn = NPT(
                atoms=ase_atoms,
                timestep=self._timestep * units.fs,
                temperature_K=self._temperature,
                externalstress=self._external_stress,
                ttime=self._ttime * units.fs,
                pfactor=None,
                mask=np.array(self._mask),
                trajectory=self.trajectory,
                logfile=self._logfile,
                loginterval=self._loginterval,
                append_trajectory=self._append_trajectory,
        )

    def _initialize_nve(self, ase_atoms) -> None:
        """
        Initializes the NVE ensemble for the MD simulation.

        Args:
            ase_atoms (Atoms): The ASE atoms object.
        """
        self.dyn = VelocityVerlet(
                atoms=ase_atoms,
                timestep=self._timestep * units.fs,
                trajectory=self.trajectory,
                logfile=self._logfile,
                loginterval=self._loginterval,
                append_trajectory=self._append_trajectory,
        )

    def calculate(self, structure: Structure) -> dict:
        """
        Performs the Molecular Dynamics (MD) simulation using the M3GNet potential.

        Args:
            structure (Structure): The input structure.

        Returns:
            dict: A dictionary containing the results of the MD simulation.
        """
        ase_atoms = structure.to_ase_atoms()
        self.trajectory = TrajectoryObserver(ase_atoms)

        MaxwellBoltzmannDistribution(ase_atoms, temperature_K=self._temperature)

        ase_atoms.calc = PESCalculator(potential=self.potential)

        if self._ensemble.lower() == "npt_nose_hoover":
            self._initialize_npt_nose_hoover(ase_atoms)

        if self._ensemble.lower() == "nvt_nose_hoover":
            self._initialize_nvt_nose_hoover(ase_atoms)

        if self._ensemble.lower() == "nve":
            self._initialize_nve(ase_atoms)

        self.dyn.run(self._steps)

        average_results = self._get_average_results()

        if average_results:
            return {
                    "energy": average_results["average_energy"],
                    "potential_energy": average_results["average_energy"],
                    "forces": average_results["average_forces"],
                    "stresses": average_results["average_stresses"],
                    "temperature": average_results["average_temperature"],
            }

        return {
                "energy": ase_atoms.get_potential_energy(),
                "potential_energy": ase_atoms.get_potential_energy(),
                "forces": ase_atoms.get_forces(),
                "stresses": ase_atoms.get_stress(),
                "temperature": ase_atoms.get_temperature(),
        }


class TrajectoryObserver(collections.abc.Sequence):
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.

    NOTE: Adapted from the matgl code and added the capability to save the temperature values.
    """

    def __init__(self, atoms: Atoms) -> None:
        """
        Initializes the TrajectoryObserver from the ase Atoms object.

        Args:
            atoms (Atoms): Structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[ArrayLike] = []
        self.stresses: list[ArrayLike] = []
        self.temperature: list[ArrayLike] = []
        self.cells: list[ArrayLike] = []
        self.atom_positions: list[ArrayLike] = []

    def __call__(self) -> None:
        """Saves the current state of the atoms."""
        self.energies.append(float(self.atoms.get_potential_energy()))
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.temperature.append(self.atoms.get_temperature())
        self.cells.append(self.atoms.get_cell()[:])
        self.atom_positions.append(self.atoms.get_positions())

    def __getitem__(self, item):
        """Returns a tuple of properties at the given index."""
        return self.energies[item], self.forces[item], self.stresses[item], self.temperature[item], self.cells[item], self.atom_positions[item]

    def __len__(self):
        """Returns the number of saved properties."""
        return len(self.energies)

    def as_pandas(self) -> pd.DataFrame:
        """
        Returns the trajectory as a pandas DataFrame
        of energies, forces, stresses, temperatures, cells and atom_positions.

        Returns:
            pd.DataFrame: The trajectory as a pandas DataFrame.
        """
        return pd.DataFrame(
                {
                        "energies": self.energies,
                        "forces": self.forces,
                        "stresses": self.stresses,
                        "temperature": self.temperature,
                        "cells": self.cells,
                        "atom_positions": self.atom_positions,
                }
        )
