"""This module provides classes and utilities for relaxing and calculating atomic structures.

`BaseCalculator` is an abstract base class that defines a common interface for structure
relaxation and calculation using various optimization algorithms.
"""

from __future__ import annotations

import contextlib
import io
import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from ase.constraints import FixAtoms, FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch, LBFGSLineSearch, MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG

from materialsframework.tools.trajectory import TrajectoryObserver
from materialsframework.utils import to_atoms, to_structure

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Molecule, Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class OPTIMIZERS(Enum):
    """Enumeration of optimization algorithms available for structure relaxation.

    Each member wraps one of the Atomic Simulation Environment (ASE) optimizers.

    Attributes:
        bfgs (BFGS): BFGS optimization method.
        bfgslinesearch (BFGSLineSearch): BFGS with line search optimization.
        fire (FIRE): Fast Inertial Relaxation Engine (FIRE) optimizer.
        lbfgs (LBFGS): Limited-memory Broyden-Fletcher-Goldfarb-Shanno optimizer.
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
    """Abstract base class for structure relaxers and calculators built on the Atomic Simulation Environment (ASE).

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
        """Abstract class-level property that must be defined in all subclasses.

        Returns:
            list[str]: Names of the properties the calculator can compute, such as
            "potential_energy", "forces", or "stress".
        """

    def __init__(
        self,
        fmax: float = 0.1,
        steps: int = 1000,
        optimizer: type[Optimizer] | str = "FIRE",
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
        **kwargs: Any,
    ):
        """Initializes the BaseCalculator with parameters for structure relaxation.

        Args:
            fmax (float, optional): Maximum force convergence criterion. Defaults to 0.1.
            steps (int, optional): Maximum number of optimization steps. Defaults to 1000.
            optimizer (type[Optimizer] | str, optional): The optimization algorithm to use. Can be
                either an Optimizer subclass or a string referring to one of the OPTIMIZERS
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
            **kwargs: Forwarded to the next class in the MRO (e.g. `BaseMDCalculator`), so cooperative
                subclasses can chain a single `super().__init__(**kwargs)` call instead of splitting
                kwargs by hand.
        """
        if not hasattr(self.__class__, "AVAILABLE_PROPERTIES"):
            raise TypeError(f"Class {self.__class__.__name__} must define AVAILABLE_PROPERTIES")

        self.fmax = fmax
        self.steps = steps
        self.optimizer: type[Optimizer] = OPTIMIZERS[optimizer.lower()].value if isinstance(optimizer, str) else optimizer
        self.relax_cell = relax_cell
        self.fix_symmetry = fix_symmetry
        self.fix_atoms = fix_atoms
        self.hydrostatic_strain = hydrostatic_strain
        self.sym_prec = symprec
        self.traj_file = traj_file
        self.interval = interval
        self.verbose = verbose
        self.params_asecellfilter = params_asecellfilter

        self.converged = None

        self.include_magmoms = include_magmoms
        self.include_dipoles = include_dipoles

        super().__init__(**kwargs)

    @property
    @abstractmethod
    def calculator(self) -> Calculator:
        """Returns the ASE Calculator object associated with this relaxer.

        Subclasses of BaseCalculator must implement this property; the returned Calculator
        object performs the relaxation and calculation of structures within `relax()`.

        Raises:
            NotImplementedError: If the subclass does not implement this property.

        Returns:
            Calculator: An ASE Calculator instance configured for the specific
            relaxation and calculation task.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'calculator' property to return a valid ASE Calculator instance."
        )

    def relax(
        self,
        structure: Atoms | Structure | Molecule,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Relaxes a given atomic structure using the specified optimizer and calculator.

        Args:
            structure (Atoms | Structure | Molecule): The atomic structure to relax. This can be an ASE `Atoms` object,
                a Pymatgen `Structure` object, or a Pymatgen `Molecule` object.
            **kwargs: Additional keyword arguments to pass to the optimizer during relaxation.

        Returns:
            dict[str, Any]: Dictionary with keys:
                - ``final_structure``: Final relaxed structure as a pymatgen ``Structure``.
                - ``trajectory``: ``TrajectoryObserver`` containing intermediate relaxation states.
                - Property keys from ``AVAILABLE_PROPERTIES`` (for example ``energy``, ``forces``, ``stress``)
                    populated from the calculator results.

        Raises:
            ValueError: If the structure cannot be relaxed.
        """
        stream = sys.stdout if self.verbose else io.StringIO()
        params_asecellfilter = self.params_asecellfilter or {}

        atoms = to_atoms(structure)

        self._reset_calculator_results()
        atoms.calc = self.calculator

        constraints = []
        if self.fix_symmetry:
            constraints.append(FixSymmetry(atoms=atoms, symprec=self.sym_prec))
        if self.fix_atoms:
            constraints.append(FixAtoms(mask=[True for _ in atoms]))
        if constraints:
            atoms.set_constraint(constraints)

        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(
                atoms,
                include_magmoms=self.include_magmoms,
                include_dipoles=self.include_dipoles,
            )
            if self.relax_cell:
                atoms = FrechetCellFilter(
                    atoms=atoms,
                    hydrostatic_strain=self.hydrostatic_strain,
                    **params_asecellfilter,
                )
            optimizer = self.optimizer(atoms, **kwargs)  # ty: ignore[invalid-argument-type]
            optimizer.attach(obs, interval=self.interval)
            optimizer.run(fmax=self.fmax, steps=self.steps)
            obs()

            self.converged = optimizer.nsteps < self.steps

        if self.traj_file:
            obs.save(self.traj_file)

        if isinstance(atoms, FrechetCellFilter):
            atoms = atoms.atoms

        out_dict = {
            "final_structure": to_structure(atoms),
            "trajectory": obs,
        }

        out_dict.update({prop: self.calculator.results.get(prop, None) for prop in self.AVAILABLE_PROPERTIES})

        return out_dict

    def calculate(
        self,
        structure: Atoms | Structure | Molecule,
    ) -> dict[str, Any]:
        """Performs a single-point calculation on the given atomic structure using the specified calculator.

        No relaxation is performed. The properties to compute are defined in the
        `AVAILABLE_PROPERTIES` class attribute.

        Args:
            structure (Atoms | Structure | Molecule): The atomic structure to calculate. This can be an ASE `Atoms` object,
                a Pymatgen `Structure` object, or a Pymatgen `Molecule` object.

        Returns:
            dict[str, Any]: Dictionary with keys:
                - ``final_structure``: Input structure as a pymatgen ``Structure``.
                - Property keys from ``AVAILABLE_PROPERTIES`` (for example ``energy``, ``forces``, ``stress``)
                    populated from the calculator results.
        """
        atoms = to_atoms(structure)

        self._reset_calculator_results()
        atoms.calc = self.calculator
        self.calculator.calculate(
            atoms=atoms,
            properties=self.AVAILABLE_PROPERTIES,
            system_changes=[
                "positions",
                "numbers",
                "cell",
                "pbc",
                "initial_charges",
                "initial_magmoms",
            ],
        )

        out_dict = {
            "final_structure": to_structure(atoms),
        }

        out_dict.update({prop: self.calculator.results[prop] for prop in self.AVAILABLE_PROPERTIES})

        return out_dict

    def _reset_calculator_results(self) -> None:
        """Clear cached calculator outputs before a new evaluation.

        Some ASE calculators cache result arrays keyed to the previous number of atoms.
        Clearing the cache avoids shape mismatches when the next structure has a different
        site count, such as interstitial defect structures.
        """
        calculator = self.calculator
        results = getattr(calculator, "results", None)
        if isinstance(results, dict):
            results.clear()
