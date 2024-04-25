"""
This module provides classes to perform calculations using the M3GNet potential.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Union

import matgl
import numpy as np
from matgl.ext.ase import PESCalculator, Relaxer as AseM3GNetRelaxer

if TYPE_CHECKING:
    from matgl.apps.pes import Potential
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure

from materialsframework.calculators.typing import Calculator, Relaxer

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class M3GNetRelaxer(Relaxer):
    """
    A class used to represent a M3GNet Relaxer.

    This class provides methods to perform relaxation of a structure using the M3GNet potential.
    """

    def __init__(
            self,
            fmax: float = 0.001,
            relax_cell: bool = True,
            verbose: bool = False,
            steps: int = 1000,
            model: str = "M3GNet-MP-2021.2.8-PES",
    ) -> None:
        """
        Initializes the M3GNet calculator.

        Args:
            fmax (float): The maximum force tolerance for convergence. Defaults to 0.001.
            relax_cell (bool): Whether to relax the lattice cell. Defaults to True.
            verbose (bool): Whether to print verbose output during calculations. Defaults to False.
            steps (int): The maximum number of optimization steps. Defaults to 1000.
            model (str): The M3GNet model to use. Defaults to "M3GNet-MP-2021.2.8-PES".

        Examples:
            >>> relaxer = M3GNetRelaxer()
            >>> relaxer = M3GNetRelaxer(fmax=0.001, relax_cell=True, verbose=False, steps=1000,
            ...                         model="M3GNet-MP-2021.2.8-PES")

        Note:
            The remaining values for the arguments are set to the default values for the M3GNet potential.
        """
        self._fmax = fmax
        self._relax_cell = relax_cell
        self._verbose = verbose
        self._model = model
        self._steps = steps

        self._relaxer = None
        self._potential = None

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

    @property
    def relaxer(self) -> AseM3GNetRelaxer:
        """
        Returns the Relaxer object associated with this instance.

        If the Relaxer object has not been initialized yet, it will be created using the
        potential and relax_cell attributes of this instance.

        Returns:
            AseRelaxer: The AseM3GNetRelaxer object associated with this instance.
        """
        if self._relaxer is None:
            self._relaxer = AseM3GNetRelaxer(potential=self.potential, relax_cell=self._relax_cell)
        return self._relaxer

    def relax(
            self,
            structure: Structure,
    ) -> tuple[Structure, float]:
        """
        Performs the relaxation of the structure using the M3GNet calculator.

        Args:
            structure (Structure): The input structure.

        Returns:
            tuple[Structure, float]: A tuple containing the relaxed structure and its energy.

        Examples:
            >>> relaxer = M3GNetRelaxer()
            >>> struct = Structure.from_file("POSCAR")
            >>> relaxation_results = relaxer.relax(structure=struct)
        """
        relax_results = self.relaxer.relax(structure, fmax=self._fmax, steps=self._steps, verbose=self._verbose)
        return {
                'final_structure': relax_results["final_structure"],
                'energy': float(relax_results["trajectory"].energies[-1])
        }


class M3GNetCalculator(Calculator):
    """
    A class representing the M3GNet calculator.

    This class is used to calculate the potential energy, forces, and stresses
    of a given structure using the M3GNet potential.
    """

    def __init__(self, model: str = "M3GNet-MP-2021.2.8-PES") -> None:
        """
        Initializes the M3GNet calculator.

        Args:
            model (str): The M3GNet model to use. Defaults to "M3GNet-MP-2021.2.8-PES".

        Examples:
            >>> calculator = M3GNetCalculator()
            >>> calculator = M3GNetCalculator(model="M3GNet-MP-2021.2.8-PES")

        Note:
            The remaining values for the arguments are set to the default values for the M3GNet potential.
        """
        self._model: str = model

        self._calculator = None
        self._potential = None

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

    @property
    def calculator(self) -> PESCalculator:
        """
        Returns the M3GNet PESCalculator instance.

        If the calculator instance is not already created, it creates a new PESCalculator instance
        with the specified potential and returns it. Otherwise, it returns the existing
        calculator instance.

        Returns:
            PESCalculator: The M3GNet calculator instance.
        """
        if self._calculator is None:
            self._calculator = PESCalculator(potential=self.potential)
        return self._calculator

    def calculate(
            self,
            structure: Structure,
    ) -> dict[str, Union[float, ArrayLike]]:
        """
        Calculates the potential energy, forces, and stresses of the given structure.

        Args:
            structure (Structure): The structure for which the properties will be calculated.

        Returns:
            dict[str, ArrayLike]: A dictionary containing the calculated properties.

        Examples:
            >>> calculator = M3GNetCalculator()
            >>> struct = Structure.from_file("POSCAR")
            >>> calculation_results = calculator.calculate(structure=struct)
        """
        atoms = structure.to_ase_atoms()
        atoms.calc = self.calculator
        return {
                "potential_energy": np.array(atoms.get_potential_energy()),
                "forces": np.array(atoms.get_forces()),
                "stresses": np.array(atoms.get_stress())
        }
