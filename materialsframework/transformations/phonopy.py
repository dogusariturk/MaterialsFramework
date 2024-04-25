"""
This module provides a class to generate distorted structures for Phonopy calculations.
"""
from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import numpy as np
from phonopy import Phonopy
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from materialsframework.calculators import Calculator, Relaxer

from materialsframework.calculators import M3GNetCalculator, M3GNetRelaxer

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class PhonopyDisplacementTransformation:
    """
    A class used to represent a Phonopy Displacement Transformation.

    This class provides methods to generate displaced structures for Phonopy calculations.
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
        Initializes the PhonopyDisplacementTransformation.

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
        self.displacements = None
        self.displaced_structures = None

    def apply_transformation(
            self,
            structure: Structure,
            distance: float = 0.01,
            supercell_matrix: Optional[list] = None,
            primitive_matrix: Optional[list] = None,
            is_relaxed: bool = False,
            log_level: int = 0,
            **kwargs
    ) -> None:
        """
        Applies the transformation to generate displaced supercells using Phonopy.

        Args:
            structure (Structure): The input structure to be displaced.
            distance (float): The maximum distance to displace the atoms. Defaults to 0.01.
            supercell_matrix (list): The supercell matrix to generate the supercells. Defaults to None.
            primitive_matrix (list): The primitive matrix to generate the primitive cell. Defaults to None.
            is_relaxed (bool): Whether the input structure is already relaxed. Defaults to False.
            log_level (int): The log level to use for Phonopy. Defaults to 0.
            **kwargs: Additional keyword arguments to pass to the Phonopy.generate_displacement method.
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

    def _get_displaced_structures(self, distance: float = 0.01, **kwargs) -> list[Structure, ...]:
        """
        This method generates displaced structures using Phonopy.

        Args:
            distance (float): The maximum distance to displace the atoms.
        Returns:
            list[Structure, ...]: A list of displaced structures.
        """
        self.phonon.generate_displacements(distance=distance, **kwargs)

        displaced_supercells = self.phonon.supercells_with_displacements
        displaced_structures = [get_pmg_structure(cell) for cell in displaced_supercells
                                if cell is not None]

        return displaced_structures
