from __future__ import annotations

import collections
import os
import pickle
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ase import Atoms
    from numpy.typing import ArrayLike

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


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
        self.total_energies: list[float] = []
        self.kinetic_energies: list[float] = []
        self.potential_energies: list[float] = []
        self.forces: list[ArrayLike] = []
        self.stresses: list[ArrayLike] = []
        self.temperature: list[ArrayLike] = []
        self.cells: list[ArrayLike] = []
        self.atom_positions: list[ArrayLike] = []

    def __call__(self) -> None:
        """Saves the current state of the atoms."""
        self.total_energies.append(float(self.atoms.get_total_energy()))
        self.kinetic_energies.append(float(self.atoms.get_kinetic_energy()))
        self.potential_energies.append(float(self.atoms.get_potential_energy()))
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.temperature.append(self.atoms.get_temperature())
        self.cells.append(self.atoms.get_cell()[:])
        self.atom_positions.append(self.atoms.get_positions())

    def __getitem__(self, item):
        """Returns a tuple of properties at the given index."""
        item_properties = (
                self.total_energies[item],
                self.kinetic_energies[item],
                self.potential_energies[item],
                self.forces[item],
                self.stresses[item],
                self.temperature[item],
                self.cells[item],
                self.atom_positions[item]
        )
        return item_properties

    def __len__(self):
        """Returns the number of saved properties."""
        return len(self.total_energies)

    def as_pandas(self) -> pd.DataFrame:
        """
        Returns the trajectory as a pandas DataFrame
        of energies, forces, stresses, temperatures, cells and atom_positions.

        Returns:
            pd.DataFrame: The trajectory as a pandas DataFrame.
        """
        return pd.DataFrame(
                {
                        "total_energies": self.total_energies,
                        "potential_energies": self.potential_energies,
                        "kinetic_energies": self.kinetic_energies,
                        "forces": self.forces,
                        "stresses": self.stresses,
                        "temperature": self.temperature,
                        "cells": self.cells,
                        "atom_positions": self.atom_positions,
                }
        )

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory.
        """
        out = {
                "total_energies": self.total_energies,
                "potential_energies": self.potential_energies,
                "kinetic_energies": self.kinetic_energies,
                "forces": self.forces,
                "stresses": self.stresses,
                "temperature": self.temperature,
                "atom_positions": self.atom_positions,
                "cell": self.cells,
                "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out, file)
