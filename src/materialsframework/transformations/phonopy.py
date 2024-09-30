"""
This module provides a class to generate distorted structures for Phonopy calculations.

The `PhonopyDisplacementTransformation` class facilitates the generation of supercells with atomic displacements,
which are required for calculating force constants and phonon properties using Phonopy. These displaced structures
are essential for studying vibrational modes, thermal properties, and lattice dynamics in materials.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from phonopy import Phonopy
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from materialsframework.calculators.m3gnet import M3GNetCalculator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.tools.calculator import BaseCalculator

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class PhonopyDisplacementTransformation:
    """
    A class used to generate displaced structures for Phonopy calculations.

    The `PhonopyDisplacementTransformation` class provides methods to create supercells with atomic
    displacements, which are necessary for calculating force constants and phonon properties. These
    supercells can be used in conjunction with Phonopy to calculate phonon spectra, thermal conductivity,
    and other lattice dynamical properties.
    """

    def __init__(
            self,
            calculator: BaseCalculator | None = None,
    ) -> None:
        """
        Initializes the `PhonopyDisplacementTransformation` object.

        Args:
            calculator (BaseCalculator | None, optional): The calculator instance to use for relaxation.
                                                             Defaults to `M3GNetCalculator`.
        """
        self._calculator = calculator

        self.phonon: Phonopy | None = None
        self.displacements: np.ndarray | list | None = None
        self.displaced_structures: list[Structure, ...] | None = None

    def apply_transformation(
            self,
            structure: Structure,
            distance: float = 0.01,
            supercell_matrix: list | None = None,
            primitive_matrix: list | None = None,
            is_relaxed: bool = False,
            log_level: int = 0,
            **kwargs
    ) -> None:
        """
        Applies the transformation to generate displaced supercells for Phonopy calculations.

        This method generates supercells with atomic displacements for phonon calculations.
        The resulting supercells are stored in the `displaced_structures` attribute, and the displacement
        vectors are stored in `displacements`.

        Args:
            structure (Structure): The input structure to be displaced.
            distance (float, optional): The maximum atomic displacement distance. Defaults to 0.01.
            supercell_matrix (list, optional): The supercell matrix to generate supercells for phonon calculations.
                                               Defaults to a 2x2x2 supercell.
            primitive_matrix (list, optional): The primitive matrix to generate the primitive cell. Defaults to None.
            is_relaxed (bool, optional): If True, the input structure is assumed to be relaxed. Defaults to False.
            log_level (int, optional): The log level for Phonopy. Defaults to 0.
            **kwargs: Additional keyword arguments for the `Phonopy.generate_displacement` method.

        Note:
            The generated displaced structures are stored in the `displaced_structures` attribute, and the
            displacement vectors are stored in `displacements`.
        """
        supercell_matrix = supercell_matrix or np.eye(3) * np.array((2, 2, 2))

        if not is_relaxed:
            structure: Structure = self._relax_structure(structure)  # type: ignore

        phonopy_structure = get_phonopy_structure(structure)

        self.phonon = Phonopy(unitcell=phonopy_structure,
                              supercell_matrix=supercell_matrix,
                              primitive_matrix=primitive_matrix,
                              log_level=log_level)

        self.displaced_structures = self._get_displaced_structures(distance=distance, **kwargs)
        self.displacements = self.phonon.displacements

    @property
    def calculator(self) -> BaseCalculator:
        """
        Returns the Calculator instance for structure relaxation.

        If the calculator instance is not already created, this method initializes a new
        `M3GNetCalculator` instance. Otherwise, it returns the existing calculator.

        Returns:
            BaseCalculator: The calculator instance used for structure relaxation.
        """
        if self._calculator is None:
            self._calculator = M3GNetCalculator()
        return self._calculator

    def _relax_structure(
            self,
            structure: Structure
    ) -> Structure:
        """
        Relaxes the input structure using the calculator.

        This method takes a pymatgen `Structure` object as input and relaxes it using the specified calculator.
        The relaxed structure is returned.

        Args:
            structure (Structure): The initial structure to be relaxed.

        Returns:
            Structure: The relaxed structure.
        """
        return self.calculator.relax(structure)["final_structure"]

    def _get_displaced_structures(
            self,
            distance: float = 0.01,
            **kwargs
    ) -> list[Structure, ...]:
        """
        Generates displaced structures using Phonopy.

        This method generates supercells with atomic displacements for phonon calculations using Phonopy.
        The displaced structures are returned as a list of `Structure` objects.

        Args:
            distance (float, optional): The maximum atomic displacement distance. Defaults to 0.01.

        Returns:
            list[Structure, ...]: A list of displaced structures for phonon calculations.
        """
        self.phonon.generate_displacements(distance=distance, **kwargs)

        displaced_supercells = self.phonon.supercells_with_displacements
        displaced_structures = [get_pmg_structure(cell) for cell in displaced_supercells
                                if cell is not None]

        return displaced_structures
