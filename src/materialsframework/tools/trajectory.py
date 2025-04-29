"""
This module provides a `TrajectoryObserver` class for observing and recording the states
of atomic structures during relaxation processes in the Atomic Simulation Environment (ASE).

The `TrajectoryObserver` class can save properties like energies, forces, stresses,
magnetic moments, dipoles, and more for each step of the relaxation.
"""
from __future__ import annotations

import collections
import pickle
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ase import Atoms
    from numpy.typing import ArrayLike

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class TrajectoryObserver(collections.abc.Sequence):
    """
    TrajectoryObserver is a class that observes and records the states of atomic structures
    during the relaxation process in ASE.

    This class acts as a hook during the relaxation process, saving intermediate structures and
    their associated properties like energies, forces, stresses, and optionally, temperatures,
    magnetic moments, and dipoles.

    Attributes:
        atoms (Atoms): The ASE Atoms object representing the structure to observe.
        include_temperature (bool): Whether to save the temperature values. Defaults to False.
        include_magmoms (bool): Whether to save the magnetic moments. Defaults to False.
        include_dipoles (bool): Whether to save the dipoles. Defaults to False.

    Note:
        This class was adapted from the matgl code and extended to include the ability
        to save additional property values.
    """
    def __init__(
            self,
            atoms: Atoms,
            include_temperature: bool = False,
            include_magmoms: bool = False,
            include_dipoles: bool = False,
    ) -> None:
        """
        Initializes the TrajectoryObserver with the ASE Atoms object and optional flags
        for recording additional properties.

        Args:
            atoms (Atoms): The ASE Atoms object representing the atomic structure to observe and record.
            include_temperature (bool, optional): If True, the observer will record the temperature
                at each step. Defaults to False.
            include_magmoms (bool, optional): If True, the observer will record the magnetic moments
                at each step. Defaults to False.
            include_dipoles (bool, optional): If True, the observer will record the dipoles
                at each step. Defaults to False.
        """
        self.atoms = atoms
        self.include_temperature = include_temperature
        self.include_magmoms = include_magmoms
        self.include_dipoles = include_dipoles

        self.total_energies: list[float] = []
        self.potential_energies: list[float] = []
        self.kinetic_energies: list[float] = []
        self.forces: list[ArrayLike] = []
        self.stresses: list[ArrayLike] = []
        self.cells: list[ArrayLike] = []
        if self.include_temperature:
            self.temperatures: list[ArrayLike] = []
        if self.include_magmoms:
            self.magmoms: list[ArrayLike] = []
        if self.include_dipoles:
            self.dipoles: list[ArrayLike] = []
        self.atom_positions: list[ArrayLike] = []
        self.atomic_numbers: list[int] = []
        self.chemical_symbols: list[str] = []

    def __call__(self) -> None:
        """
        Records the current state of the atoms, including energies, forces, stresses,
        and optionally, temperatures, magnetic moments, and dipoles.

        This method captures and stores various properties of the ASE Atoms object at the current
        step of the relaxation process.
        """
        self.total_energies.append(float(self.atoms.get_total_energy()))
        self.potential_energies.append(float(self.atoms.get_potential_energy()))
        self.kinetic_energies.append(float(self.atoms.get_kinetic_energy()))
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.cells.append(self.atoms.get_cell()[:])
        if self.include_temperature:
            self.temperatures.append(self.atoms.get_temperature())
        if self.include_magmoms:
            self.magmoms.append(self.atoms.get_magnetic_moments())
        if self.include_dipoles:
            self.dipoles.append(self.atoms.get_array("dipole"))
        self.atom_positions.append(self.atoms.get_positions())
        self.atomic_numbers.append(self.atoms.get_atomic_numbers())
        self.chemical_symbols.append(self.atoms.get_chemical_symbols())

    def __getitem__(self, item):
        """
        Returns a tuple of recorded properties at the specified index.

        Args:
            item (int): The index of the step to retrieve properties for.

        Returns:
            tuple: A tuple containing the total energies, potential energies, kinetic energies,
            forces, stresses, cell parameters, atomic positions, atomic numbers, chemical symbols,
            and, if applicable, temperatures, magnetic moments, and dipoles at the specified step.
        """
        item_properties = (
                self.total_energies[item],
                self.potential_energies[item],
                self.kinetic_energies[item],
                self.forces[item],
                self.stresses[item],
                self.cells[item],
                self.atom_positions[item],
                self.atomic_numbers[item],
                self.chemical_symbols[item],
        )
        if self.include_temperature:
            item_properties += self.temperatures[item],
        if self.include_magmoms:
            item_properties += self.magmoms[item],
        if self.include_dipoles:
            item_properties += self.dipoles[item],
        return item_properties

    def __len__(self):
        """
        Returns the number of recorded steps in the observer.

        This method provides the length of the recorded trajectory, corresponding to the
        number of steps at which properties were saved.
        """

        return len(self.total_energies)

    def _out_dict(self) -> dict[str, Any]:
        """
        Returns a dictionary containing all recorded properties from the relaxation process.

        Returns:
            dict[str, Any]: A dictionary with the following keys:
                - "total_energies": List of total energies recorded at each step.
                - "potential_energies": List of potential energies recorded at each step.
                - "kinetic_energies": List of kinetic energies recorded at each step.
                - "forces": List of forces on atoms recorded at each step.
                - "stresses": List of stresses on the structure recorded at each step.
                - "cells": List of cell parameters recorded at each step.
                - "atom_positions": List of atomic positions recorded at each step.
                - "atomic_numbers": List of atomic numbers recorded at each step.
                - "chemical_symbols": List of chemical symbols recorded at each step.
                - "temperatures" (optional): List of temperatures recorded at each step, if applicable.
                - "magmoms" (optional): List of magnetic moments recorded at each step, if applicable.
                - "dipoles" (optional): List of dipoles recorded at each step, if applicable.
        """
        out_dict = {
                "total_energies": self.total_energies,
                "potential_energies": self.potential_energies,
                "kinetic_energies": self.kinetic_energies,
                "forces": self.forces,
                "stresses": self.stresses,
                "cells": self.cells,
                "atom_positions": self.atom_positions,
                "atomic_numbers": self.atomic_numbers,
                "chemical_symbols": self.chemical_symbols,
        }
        if self.include_temperature:
            out_dict["temperatures"] = self.temperatures
        if self.include_magmoms:
            out_dict["magmoms"] = self.magmoms
        if self.include_dipoles:
            out_dict["dipoles"] = self.dipoles
        return out_dict

    def as_pandas(self) -> pd.DataFrame:
        """
        Converts the recorded trajectory into a pandas DataFrame.

        The DataFrame will contain columns for total energies, potential energies,
        kinetic energies, forces, stresses, cell parameters, atomic positions,
        atomic numbers, chemical symbols, and, if applicable, temperatures, magnetic moments, and dipoles.

        Returns:
            pd.DataFrame: A DataFrame where each row corresponds to a step in the trajectory,
            and each column corresponds to a recorded property.
        """
        return pd.DataFrame(self._out_dict())

    def save(self, filename: str) -> None:
        """
        Saves the recorded trajectory to a file in binary format using pickle.

        Args:
            filename (str): The name of the file where the trajectory will be saved.

        The trajectory data, including all recorded properties, will be serialized
        and saved to the specified file.
        """
        with open(filename, "wb") as file:
            pickle.dump(self._out_dict(), file)
