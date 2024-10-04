"""
This module provides a class to generate distorted structures for Phono3py calculations.

The `Phono3pyDisplacementTransformation` class facilitates the generation of supercells with atomic displacements,
which are necessary for calculating second- and third-order force constants using Phono3py. These displaced structures
are critical in studying anharmonic phonon properties and thermal conductivity in materials.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from phono3py import Phono3py
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from materialsframework.calculators.m3gnet import M3GNetCalculator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class Phono3pyDisplacementTransformation:
    """
    A class used to generate displaced structures for Phono3py calculations.

    The `Phono3pyDisplacementTransformation` class provides methods to create supercells with atomic
    displacements needed for Phono3py calculations. It supports the generation of structures for both
    second- and third-order force constants, which are crucial for phonon calculations, including the
    study of lattice thermal conductivity and phonon interactions.
    """

    def __init__(
            self,
            calculator: BaseCalculator| None = None,
    ) -> None:
        """
        Initializes the `Phono3pyDisplacementTransformation` object.

        Args:
            calculator (BaseCalculator | None, optional): The calculator instance to use for relaxation.
                                                             Defaults to `M3GNetCalculator`.
        """
        self._calculator = calculator

        self.phonon: Phono3py | None = None

        # For second-order force-constant results
        self.phonon_displacements = None
        self.phonon_supercells_with_displacements = None

        # For third-order force-constant results
        self.supercell_displacements = None
        self.supercells_with_displacements = None

    def apply_transformation(
            self,
            structure: Structure,
            distance: float = 0.03,
            supercell_matrix: list | None = None,
            primitive_matrix: list | None = None,
            phonon_supercell_matrix: list | None = None,
            is_relaxed: bool = False,
            log_level: int = 0,
            **kwargs
    ) -> None:
        """
        Applies the transformation to generate displaced supercells for Phono3py calculations.

        This method generates supercells with atomic displacements for both second-order (phonon) and
        third-order force constant calculations. These supercells are necessary for calculating phonon
        properties and investigating lattice dynamics.

        Args:
            structure (Structure): The input structure to be used for generating displacements.
            distance (float, optional): The maximum atomic displacement distance. Defaults to 0.03.
            supercell_matrix (list, optional): The supercell matrix for third-order force constant calculations.
                                               Defaults to a 2x2x2 supercell.
            primitive_matrix (list, optional): The primitive matrix for second-order force constant calculations.
                                               Defaults to None.
            phonon_supercell_matrix (list, optional): The supercell matrix for second-order force constant calculations.
                                                      Defaults to a 3x3x3 supercell.
            is_relaxed (bool, optional): If True, the input structure is assumed to be relaxed. Defaults to False.
            log_level (int, optional): The log level for Phono3py. Defaults to 0.
            **kwargs: Additional keyword arguments for the `Phono3py.generate_displacement` method.

        Note:
            The generated displaced supercells are stored in `phonon_supercells_with_displacements` (for phonon calculations)
            and `supercells_with_displacements` (for third-order force constants).
        """
        supercell_matrix = supercell_matrix or np.eye(3) * np.array((2, 2, 2))
        phonon_supercell_matrix = phonon_supercell_matrix or np.eye(3) * np.array((3, 3, 3))

        if not is_relaxed:
            structure: Structure = self._relax_structure(structure)  # type: ignore

        phonopy_structure = get_phonopy_structure(structure)

        self.phonon = Phono3py(unitcell=phonopy_structure,
                               supercell_matrix=supercell_matrix,
                               primitive_matrix=primitive_matrix,
                               phonon_supercell_matrix=phonon_supercell_matrix,
                               log_level=log_level)

        self.phonon_supercells_with_displacements, self.supercells_with_displacements = self._get_displaced_structures(
                distance=distance, **kwargs)

        self.phonon_displacements = self.phonon.phonon_displacements
        self.supercell_displacements = self.phonon.displacements

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

    def _relax_structure(self, structure: Structure) -> Structure:
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
            distance: float = 0.03,
            is_plusminus: bool | str = "auto",
            is_diagonal: bool = True
    ) -> tuple[list[Structure, ...], list[Structure, ...]]:
        """
        Generates displaced structures using Phono3py.

        This method generates the necessary supercells with atomic displacements for Phono3py calculations
        by applying specified displacement distances.

        Args:
            distance (float, optional): The maximum atomic displacement distance. Defaults to 0.03.
            is_plusminus (bool | str, optional): Whether to generate both positive and negative displacements.
                                                 Defaults to "auto".
            is_diagonal (bool, optional): Whether to only displace atoms along diagonal directions. Defaults to True.

        Returns:
            tuple[list[Structure, ...], list[Structure, ...]]: Two lists of displaced structures for phonon (second-order)
                                                               and third-order force constant calculations.
        """
        self.phonon.generate_displacements(distance=distance,
                                           is_plusminus=is_plusminus,
                                           is_diagonal=is_diagonal)

        displaced_supercells = self.phonon.supercells_with_displacements
        displaced_structures = [get_pmg_structure(cell) for cell in displaced_supercells
                                if cell is not None]

        self.phonon.generate_fc2_displacements(distance=distance,
                                               is_plusminus=is_plusminus,
                                               is_diagonal=is_diagonal)

        displaced_phonon_supercells = self.phonon.phonon_supercells_with_displacements
        displaced_phonon_structures = [get_pmg_structure(cell) for cell in displaced_phonon_supercells
                                       if cell is not None]

        return displaced_phonon_structures, displaced_structures
