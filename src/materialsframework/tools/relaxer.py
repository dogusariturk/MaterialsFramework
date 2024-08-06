from __future__ import annotations

import contextlib
import io
import sys
from enum import Enum
from typing import Literal, TYPE_CHECKING

from ase.constraints import FixSymmetry
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
    """An enumeration of optimizers for used in."""
    bfgs = BFGS
    bfgslinesearch = BFGSLineSearch
    fire = FIRE
    lbfgs = LBFGS
    lbfgslinesearch = LBFGSLineSearch
    mdmin = MDMin
    scipyfminbfgs = SciPyFminBFGS
    scipyfmincg = SciPyFminCG


class Relaxer:
    def __init__(
            self,
            potential: Potential,
            state_attr: torch.Tensor | None = None,
            optimizer: Optimizer | str = "FIRE",
            relax_cell: bool = True,
            fix_symmetry: bool = False,
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
        self.sym_prec = symprec
        self.potential = potential
        self.ase_adaptor = AseAtomsAdaptor()

    def relax(
            self,
            atoms: Atoms | Structure | Molecule,
            fmax: float = 0.1,
            steps: int = 500,
            traj_file: str | None = None,
            interval: int = 1,
            verbose: bool = False,
            ase_cellfilter: Literal["Frechet", "Exp"] = "Frechet",
            params_asecellfilter: dict | None = None,
            **kwargs,
    ):
        """
        Relax an input Atoms.

        Args:
            atoms (Atoms | Structure | Molecule): the atoms for relaxation
            fmax (float): total force tolerance for relaxation convergence.
            Here fmax is a sum of force and stress forces
            steps (int): max number of steps for relaxation
            traj_file (str): the trajectory file for saving
            interval (int): the step interval for saving the trajectories
            verbose (bool): Whether to have verbose output.
            ase_cellfilter (literal): which filter is used for variable cell relaxation. Default is Frechet.
            params_asecellfilter (dict): Parameters to be passed to FrechetCellFilter. Allows
                setting of constant pressure or constant volume relaxations, for example. Refer to
                https://wiki.fysik.dtu.dk/ase/ase/filters.html#FrechetCellFilter for more information.
            **kwargs: Kwargs pass-through to optimizer.
        """
        if isinstance(atoms, (Structure, Molecule)):
            atoms = self.ase_adaptor.get_atoms(atoms)
        atoms.calc = self.calculator
        if self.fix_symmetry:
            atoms.set_constraint([FixSymmetry(atoms=atoms, symprec=self.sym_prec)])
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
