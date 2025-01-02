"""
This module provides classes and utilities for relaxing and calculating atomic structures
using the Atomic Simulation Environment (ASE).

The `BaseCalculator` class serves as an abstract base class that defines a common
interface for structure relaxation and calculation using various optimization algorithms.
"""
from __future__ import annotations

import contextlib
import io
import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, BFGSLineSearch, FIRE, LBFGS, LBFGSLineSearch, MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from materialsframework.tools.trajectory import TrajectoryObserver

if TYPE_CHECKING:
    from ase import Atoms
    from numpy.typing import ArrayLike
    from ase.optimize.optimize import Optimizer

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class OPTIMIZERS(Enum):
    """
    Enumeration of available optimizers for structure relaxation.

    Each member of this enum represents a specific optimization algorithm
    from the Atomic Simulation Environment (ASE) that can be used to relax
    atomic structures.

    Attributes:
        bfgs (BFGS): BFGS optimization method.
        bfgslinesearch (BFGSLineSearch): BFGS with line search optimization.
        fire (FIRE): Fast Inertial Relaxation Engine (FIRE) optimizer.
        lbfgs (LBFGS): Limited-memory Broyden–Fletcher–Goldfarb–Shanno optimizer.
        lbfgslinesearch (LBFGSLineSearch): LBFGS with line search optimization.
        mdmin (MDMin): Molecular dynamics minimization.
        scipyfminbfgs (SciPyFminBFGS): BFGS optimization using SciPy.
        scipyfmincg (SciPyFminCG): Conjugate gradient optimization using SciPy.
    """
    bfgs = BFGS
    bfgslinesearch = BFGSLineSearch
    fire = FIRE
    lbfgs = LBFGS
    lbfgslinesearch = LBFGSLineSearch
    mdmin = MDMin
    scipyfminbfgs = SciPyFminBFGS
    scipyfmincg = SciPyFminCG


class BaseCalculator(ABC):
    """
    Abstract base class for implementing structure relaxers and calculators using the Atomic Simulation Environment (ASE).

    This class defines the common interface and basic functionality for performing
    structure relaxation and calculations using different optimization algorithms and potentials.

    Subclasses must implement the `AVAILABLE_PROPERTIES` class attribute and the `calculator` property.

    Attributes:
        fmax (float): Maximum force convergence criterion for relaxation.
        steps (int): Maximum number of optimization steps.
        optimizer (Optimizer): The optimization algorithm used for relaxation.
        relax_cell (bool): Whether to relax the cell during optimization.
        fix_symmetry (bool): Whether to enforce symmetry constraints during relaxation.
        fix_atoms (bool): Whether to fix the positions of atoms during relaxation.
        hydrostatic_strain (bool): Whether to apply hydrostatic strain during relaxation.
        sym_prec (float): Symmetry precision used when applying symmetry constraints.
        ase_adaptor (AseAtomsAdaptor): Pymatgen-ASE converter for structure handling.
        traj_file (str or None): Path to the trajectory file where the relaxation path will be saved.
        interval (int): Frequency of recording trajectory steps.
        verbose (bool): If True, prints detailed output during relaxation.
        params_asecellfilter (dict or None): Additional parameters for ASE cell filter.
        include_magmoms (bool): Whether to include magnetic moments in the trajectory.
        include_dipoles (bool): Whether to include dipoles in the trajectory.
    """

    @property
    @classmethod
    @abstractmethod
    def AVAILABLE_PROPERTIES(cls) -> list[str]:
        """
        Abstract class-level property that must be defined in all subclasses.

        This property should return a list of strings representing the
        available properties that the calculator can compute, such as
        "potential_energy", "forces", or "stress".

        Returns:
            list[str]: A list of property names available from the calculator.
        """

    def __init__(
            self,
            fmax: float = 0.1,
            steps: int = 1000,
            optimizer: Optimizer | str = "FIRE",
            relax_cell: bool = True,
            fix_symmetry: bool = False,
            fix_atoms: bool = False,
            hydrostatic_strain: bool = False,
            symprec: float = 1e-2,
            traj_file: str | None = None,
            interval: int = 1,
            verbose: bool = False,
            params_asecellfilter: dict | None = None,
            include_magmoms: bool = False,
            include_dipoles: bool = False,
    ):
        """
        Initializes the BaseCalculator with parameters for structure relaxation.

        Args:
            fmax (float, optional): Maximum force convergence criterion. Defaults to 0.1.
            steps (int, optional): Maximum number of optimization steps. Defaults to 1000.
            optimizer (Optimizer | str, optional): The optimization algorithm to use. Can be
                either an instance of `Optimizer` or a string referring to one of the OPTIMIZERS
                enum members. Defaults to "FIRE".
            relax_cell (bool, optional): If True, relaxes the unit cell dimensions. Defaults to True.
            fix_symmetry (bool, optional): If True, enforces symmetry constraints during relaxation. Defaults to False.
            fix_atoms (bool, optional): If True, fixes the positions of all atoms during relaxation. Defaults to False.
            hydrostatic_strain (bool, optional): If True, applies hydrostatic strain during cell relaxation. Defaults to False.
            symprec (float, optional): Symmetry precision for enforcing symmetry constraints. Defaults to 1e-2.
            traj_file (str or None, optional): Path to save the trajectory file. If None, trajectory is not saved. Defaults to None.
            interval (int, optional): Interval at which trajectory is recorded. Defaults to 1.
            verbose (bool, optional): If True, prints detailed output during relaxation. Defaults to False.
            params_asecellfilter (dict or None, optional): Additional parameters for the ASE cell filter. Defaults to None.
            include_magmoms (bool, optional): If True, includes magnetic moments in the trajectory. Defaults to False.
            include_dipoles (bool, optional): If True, includes dipoles in the trajectory. Defaults to False.
        """

        if not hasattr(self.__class__, "AVAILABLE_PROPERTIES"):
            raise TypeError(f"Class {self.__class__.__name__} must define AVAILABLE_PROPERTIES")

        self.fmax = fmax
        self.steps = steps
        self.optimizer: Optimizer = OPTIMIZERS[optimizer.lower()].value if isinstance(optimizer, str) else optimizer
        self.relax_cell = relax_cell
        self.fix_symmetry = fix_symmetry
        self.fix_atoms = fix_atoms
        self.hydrostatic_strain = hydrostatic_strain
        self.sym_prec = symprec
        self.ase_adaptor = AseAtomsAdaptor()
        self.traj_file = traj_file
        self.interval = interval
        self.verbose = verbose
        self.params_asecellfilter = params_asecellfilter

        self.converged = None

        self.include_magmoms = include_magmoms
        self.include_dipoles = include_dipoles

    @property
    def calculator(self) -> Calculator:
        """
        Returns the ASE Calculator object associated with this relaxer.

        This property must be implemented in subclasses of BaseCalculator.
        The returned Calculator object is used to perform the relaxation
        and calculation of structures within the relax method.

        Raises:
            NotImplementedError: If the subclass does not implement this property.

        Returns:
            Calculator: An ASE Calculator instance configured for the specific
            relaxation and calculation task.
        """
        raise NotImplementedError("Subclasses must implement the 'calculator' property to return a valid ASE Calculator instance.")

    def relax(
            self,
            structure: Atoms | Structure | Molecule,
            **kwargs,
    ) -> dict[Structure, float]:
        """
        Performs the relaxation of a given atomic structure using the specified optimizer and calculator.

        This method relaxes the input structure (which can be an `Atoms`, `Structure`, or `Molecule` object)
        according to the settings defined in the class.

        Args:
            structure (Atoms | Structure | Molecule): The atomic structure to relax. This can be an ASE `Atoms` object,
                a Pymatgen `Structure` object, or a Pymatgen `Molecule` object.
            **kwargs: Additional keyword arguments to pass to the optimizer during relaxation.

        Returns:
            dict: A dictionary containing the final relaxed structure, the trajectory of the relaxation process,
            and any computed properties listed in `AVAILABLE_PROPERTIES`.

            Keys in the dictionary:
                - "final_structure" (Structure): The final relaxed structure.
                - "trajectory" (TrajectoryObserver): The recorded trajectory of the relaxation process.
                - Other keys corresponding to properties in `AVAILABLE_PROPERTIES`, each containing the respective value
                  from the calculator's results.

        Raises:
            ValueError: If the structure cannot be relaxed.
        """

        stream = sys.stdout if self.verbose else io.StringIO()
        params_asecellfilter = self.params_asecellfilter or {}

        atoms = structure.copy()

        if isinstance(atoms, (Structure, Molecule)):
            atoms = self.ase_adaptor.get_atoms(atoms)

        atoms.calc = self.calculator

        if self.fix_symmetry:
            atoms.set_constraint([FixSymmetry(atoms=atoms, symprec=self.sym_prec)])
        if self.fix_atoms:
            atoms.set_constraint([FixAtoms(mask=[True for _ in atoms])])

        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(
                    atoms,
                    include_magmoms=self.include_magmoms,
                    include_dipoles=self.include_dipoles
            )
            if self.relax_cell:
                atoms = FrechetCellFilter(
                        atoms=atoms,
                        hydrostatic_strain=self.hydrostatic_strain,
                        **params_asecellfilter
                )
            optimizer = self.optimizer(atoms, **kwargs)
            optimizer.attach(obs, interval=self.interval)
            optimizer.run(fmax=self.fmax, steps=self.steps)
            obs()

            self.converged = optimizer.nsteps < self.steps

        if self.traj_file:
            obs.save(self.traj_file)

        if isinstance(atoms, FrechetCellFilter):
            atoms = atoms.atoms

        out_dict = {
                "final_structure": self.ase_adaptor.get_structure(atoms),
                "trajectory": obs,
        }

        out_dict.update({
                prop: self.calculator.results.get(prop, None)
                for prop in self.__class__.AVAILABLE_PROPERTIES
        })

        return out_dict

    def calculate(
            self,
            structure: Atoms | Structure | Molecule,
    ) -> dict[Structure, float | ArrayLike]:
        """
        Performs a single-point calculation on the given atomic structure using the specified calculator.

        This method calculates the properties of the input structure (which can be an `Atoms`, `Structure`, or `Molecule` object)
        without performing any relaxation. The properties to be calculated are defined in the `AVAILABLE_PROPERTIES` class attribute.

        Args:
            structure (Atoms | Structure | Molecule): The atomic structure to calculate. This can be an ASE `Atoms` object,
                a Pymatgen `Structure` object, or a Pymatgen `Molecule` object.

        Returns:
            dict: A dictionary containing the calculated structure and properties.

            Keys in the dictionary:
                - "final_structure" (Structure): The input structure as an ASE `Structure` object.
                - Other keys corresponding to properties in `AVAILABLE_PROPERTIES`, each containing the respective value
                  from the calculator's results.
        """
        atoms = structure.copy()

        if isinstance(atoms, (Structure, Molecule)):
            atoms = self.ase_adaptor.get_atoms(atoms)

        atoms.calc = self.calculator
        self.calculator.calculate(atoms=atoms)

        out_dict = {
                "final_structure": self.ase_adaptor.get_structure(atoms),
        }

        out_dict.update({
                prop: self.calculator.results[prop]
                for prop in self.__class__.AVAILABLE_PROPERTIES
        })

        return out_dict
