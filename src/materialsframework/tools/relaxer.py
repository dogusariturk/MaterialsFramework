"""
This module provides a Relaxer class for relaxing structures using ASE and M3GNet.
"""
from __future__ import annotations

import contextlib
import io
import sys
from enum import Enum
from typing import Literal, TYPE_CHECKING

from ase.constraints import FixAtoms, FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, BFGSLineSearch, FIRE, LBFGS, LBFGSLineSearch, MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from matgl.ext.ase import PESCalculator
from pymatgen.io.ase import AseAtomsAdaptor

from materialsframework.tools.trajectory import TrajectoryObserver

if TYPE_CHECKING:
    import torch
    from ase import Atoms
    from pymatgen.core import Structure, Molecule
    from ase.optimize.optimize import Optimizer
    from matgl.apps.pes import Potential

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class OPTIMIZERS(Enum):
    """
    Enum class for available optimizers.
    """
    bfgs = BFGS
    bfgslinesearch = BFGSLineSearch
    fire = FIRE
    lbfgs = LBFGS
    lbfgslinesearch = LBFGSLineSearch
    mdmin = MDMin
    scipyfminbfgs = SciPyFminBFGS
    scipyfmincg = SciPyFminCG


class Relaxer:
    """
    A class used to represent a Relaxer.

    This class provides a method to perform relaxation of a structure using a M3GNet potential.
    """
    def __init__(
            self,
            potential: Potential,
            state_attr: torch.Tensor | None = None,
            optimizer: Optimizer | str = "FIRE",
            relax_cell: bool = True,
            fix_symmetry: bool = False,
            fix_atoms: bool = False,
            symprec: float = 1e-2,
            stress_weight: float = 1 / 160.21766208,
    ):
        self.optimizer: Optimizer = OPTIMIZERS[optimizer.lower()].value if isinstance(optimizer, str) else optimizer
        self.calculator = PESCalculator(
                potential=potential,
                state_attr=state_attr,
                stress_weight=stress_weight,  # type: ignore
        )
        self.relax_cell = relax_cell
        self.fix_symmetry = fix_symmetry
        self.fix_atoms = fix_atoms
        self.sym_prec = symprec
        self.potential = potential
        self.ase_adaptor = AseAtomsAdaptor()

    def relax(
            self,
            atoms: Atoms | Structure | Molecule,
            fmax: float = 0.1,
            steps: int = 1000,
            traj_file: str | None = None,
            interval: int = 1,
            verbose: bool = False,
            params_asecellfilter: dict | None = None,
            **kwargs,
    ):
        """
        Perform relaxation of the given atoms using the M3GNet potential.

        Args:
            atoms (Atoms | Structure | Molecule): The structure to relax.
            fmax (float): The maximum force tolerance for convergence. Defaults to 0.1.
            steps (int): The maximum number of optimization steps. Defaults to 1000.
            traj_file (str): The file to save the trajectory to. Defaults to None.
            interval (int): The interval to save the trajectory. Defaults to 1.
            verbose (bool): Whether to print verbose output during calculations. Defaults to False.
            params_asecellfilter (dict): The parameters to pass to the FrechetCellFilter. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the optimizer.
        """
        if isinstance(atoms, (Structure, Molecule)):
            atoms = self.ase_adaptor.get_atoms(atoms)
        atoms.calc = self.calculator
        if self.fix_symmetry:
            atoms.set_constraint([FixSymmetry(atoms=atoms, symprec=self.sym_prec)])
        if self.fix_atoms:
            atoms.set_constraint([FixAtoms(mask=[True for _ in atoms])])
        stream = sys.stdout if verbose else io.StringIO()
        params_asecellfilter = params_asecellfilter or {}
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if self.relax_cell:
                atoms = FrechetCellFilter(atoms, **params_asecellfilter)
            optimizer = self.optimizer(atoms, **kwargs)
            optimizer.attach(obs, interval=interval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()
        if traj_file:
            obs.save(traj_file)
        if isinstance(atoms, FrechetCellFilter):
            atoms = atoms.atoms

        return {
                "final_structure": self.ase_adaptor.get_structure(atoms),
                "trajectory": obs,
        }
