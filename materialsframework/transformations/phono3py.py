"""
This module provides a class to generate distorted structures for Phono3py calculations.
"""
from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import numpy as np
from phono3py import Phono3py
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from materialsframework.calculators import M3GNetCalculator, M3GNetRelaxer

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.calculators.typing import Calculator, Relaxer

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class Phono3pyDisplacementTransformation:
    """
    A class used to represent a Phono3py Displacement Transformation.

    This class provides methods to generate displaced structures for Phono3py calculations.
    """

    def __init__(
            self,
            fmax: float = 1e-5,
            relax_cell: bool = True,
            verbose: bool = False,
            steps: int = 1000,
            model: str = "M3GNet-MP-2021.2.8-PES"
    ) -> None:
        """
        Initializes the Phono3pyDisplacementTransformation.

        Args:
            fmax (float): The maximum force tolerance for relaxation. Defaults to 1e-5.
            relax_cell (bool): Whether to relax the lattice cell. Defaults to True.
            verbose (bool): Whether to print verbose output. Defaults to False.
            steps (int): The maximum number of relaxation steps. Defaults to 1000.
            model (str): The potential model to use for relaxation. Defaults to "M3GNet-MP-2021.2.8-PES".
        """
        self._fmax = fmax
        self._relax_cell = relax_cell
        self._verbose = verbose
        self._steps = steps
        self._model = model

        self._relaxer = None
        self._calculator = None

        self.phonon = None

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
            supercell_matrix: Optional[list] = None,
            primitive_matrix: Optional[list] = None,
            phonon_supercell_matrix: Optional[list] = None,
            is_relaxed: bool = False,
            log_level: int = 0,
            **kwargs
    ) -> None:
        """
        Applies the transformation to generate displaced supercells using Phono3py.

        Args:
            structure (Structure): The input structure to be displaced.
            distance (float): The maximum distance to displace the atoms. Defaults to 0.03.
            supercell_matrix (list): Supercell matrix used for second-order force constant calculations.
            primitive_matrix (list): Primitive matrix used for second-order force constant calculations.
            phonon_supercell_matrix (list): Supercell matrix used for third-order force constant calculations.
            is_relaxed (bool): Whether the input structure is already relaxed. Defaults to False.
            log_level (int): The log level to use for Phono3py. Defaults to 0.
            **kwargs: Additional keyword arguments to pass to the Phono3py.generate_displacement method.
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
    def relaxer(self) -> Relaxer:
        """
        Returns the Relaxer instance.

        If the relaxer instance is not already created, it creates a new M3GNetRelaxer instance
        and returns it. Otherwise, it returns the existing relaxer instance.

        Returns:
            Relaxer: The Relaxer instance.
        """
        if self._relaxer is None:
            self._relaxer = M3GNetRelaxer(fmax=self._fmax,
                                          relax_cell=self._relax_cell,
                                          verbose=self._verbose,
                                          steps=self._steps,
                                          model=self._model)
        return self._relaxer

    @property
    def calculator(self) -> Calculator:
        """
        Returns the Calculator instance.

        If the calculator instance is not already created, it creates a new M3GNetCalculator
        instance with the specified potential and returns it. Otherwise, it returns the existing
        calculator instance.

        Returns:
            Calculator: The Calculator instance.
        """
        if self._calculator is None:
            self._calculator = M3GNetCalculator(model=self._model)
        return self._calculator

    def _relax_structure(self, structure: Structure) -> Structure:
        """
        This method takes a pymatgen Structure object as input and returns a relaxed structure.
        The relaxation is performed using the M3GNetRelaxer instance associated with the class.

        Args:
            structure (Structure): The initial pymatgen Structure object that needs to be relaxed.

        Returns:
            Structure: The relaxed pymatgen Structure object.
        """
        return self.relaxer.relax(structure)['final_structure']

    def _get_displaced_structures(
            self,
            distance: float = 0.03,
            is_plusminus: bool | str = "auto",
            is_diagonal: bool = True
    ) -> tuple[list[Structure, ...], list[Structure, ...]]:
        """
        This method generates displaced structures using Phono3py.

        Args:
            distance (float): The maximum distance to displace the atoms.
        Returns:
            tuple[list[Structure, ...], list[Structure, ...]]: Two lists of displaced structures.
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
