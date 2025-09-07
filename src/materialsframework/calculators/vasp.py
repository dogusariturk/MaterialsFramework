"""This module provides a class for performing calculations and structure relaxation using VASP via ASE.

The `VASPCalculator` class is designed to calculate properties such as potential energy, forces,
stresses, and to perform structure relaxation using a standard VASP installation through ASE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pymatgen.core import Molecule, Structure

from materialsframework.tools.calculator import BaseCalculator

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator  # ASE base type


__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class VASPCalculator(BaseCalculator):
    """A calculator class for performing material property calculations and relaxations using VASP via ASE.

    The `VASPCalculator` configures INCAR/KPOINTS/POTCAR settings through ASE’s `Vasp`
    calculator and exposes a lazy `calculator` property that constructs the underlying
    ASE calculator on first use.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): Properties this calculator can request/return:
            "energy", "free_energy", "forces", "dipole", "fermi", "stress", "magmom", "magmoms".

    Notes:
        - Requires a working, licensed VASP installation.
        - You can pass the VASP launch string via the `command` argument, or set one of the
          environment variables `ASE_VASP_COMMAND`, `VASP_COMMAND`, or `VASP_SCRIPT`.
        - ASE expects pseudopotentials under `VASP_PP_PATH` containing subfolders like
          `potpaw`, `potpaw_GGA`, and `potpaw_PBE`.

    References:
        - VASP documentation: https://www.vasp.at/wiki/index.php
        - ASE VASP calculator: https://ase-lib.org/ase/calculators/vasp.html
    """

    AVAILABLE_PROPERTIES = [
        "energy",
        "free_energy",
        "forces",
        "stress",
        "magmom",
        "magmoms",
    ]

    def __init__(
        self,
        *,
        command: str | None = None,
        directory: str | None = None,
        prec: str | None = None,
        pp: str | None = None,
        kpts: Any | None = None,
        xc: str | None = None,
        encut: float | None = None,
        ismear: int | None = None,
        sigma: float | None = None,
        ediff: float | None = None,
        ediffg: float | None = None,
        ibrion: int | None = None,
        isif: int | None = None,
        nsw: int | None = None,
        nelm: int | None = None,
        ispin: int | None = None,
        setups: Any | None = None,
        lreal: str | None = None,
        ncore: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the VASPCalculator.

        Args:
            command: Launch string for VASP, e.g. "mpirun -np 8 vasp_std".
                     If omitted, ASE will use $ASE_VASP_COMMAND or $VASP_COMMAND or $VASP_SCRIPT.
            directory: Working directory for the calculation (where ASE writes inputs and runs VASP).
            prec: VASP Precision setting, e.g. "Normal", "Accurate", "Single".
            pp: Pseudopotential *set* selector (e.g., "PBE", "PW91", "LDA"), which maps to
                subdirectories under $VASP_PP_PATH (potpaw_PBE/, potpaw_GGA/, potpaw/).
                You can also pass a specific folder name present on $VASP_PP_PATH.
            kpts: K-point sampling. Options:
                  - scalar int/float: VASP “Automatic” scheme via a length cutoff,
                  - sequence [nx, ny, nz]: Monkhorst–Pack mesh (Gamma-centered if gamma=True),
                  - explicit (n, 3) or (n, 4) array of points (optional weights in 4th column).
            xc: Exchange-correlation functional, e.g. "pbe", "pbesol", "lda", etc.
            encut: Plane-wave energy cutoff in eV.
            ismear: Smearing method, e.g., -5 (tetrahedron w/ Blöchl corrections), -1 (Fermi), 0 (Gaussian), 1 (Methfessel-Paxton), etc.
            sigma: Smearing width (eV) for Gaussian/Methfessel–Paxton.
            ediff: Electronic (SCF) energy convergence threshold in eV.
            ediffg: Force convergence criterion in eV/Å.
            ibrion: Ionic relaxation algorithm, e.g., -1 (no update), 0 (molecular dynamics), 1 (RMM-DIIS), 2 (conjugate gradient), etc.
            isif: Stress calculation and relaxation settings, e.g., 2 (relax ions, keep cell fixed), 3 (relax ions and cell shape), etc.
            nsw: Number of ionic steps for relaxation.
            nelm: Maximum number of electronic steps per SCF loop.
            ispin: Spin polarization setting, e.g., 1 (non-spin polarized), 2 (spin polarized).
            setups: Pseudopotential setups, e.g. "minimal", "recommended", "materialsproject", etc.
            lreal: Real-space projection setting, e.g. "Auto", "True", "False".
            ncore: Parallelization knob controlling VASP workload distribution (see VASP manual).
            **kwargs: Additional keyword arguments passed to `BaseCalculator`
                      and any remaining to `ase.calculators.vasp.Vasp`. Common VASP/INCAR/KPOINTS options exposed for convenience.
                      Any of these left as None are simply not written—VASP defaults apply. (ASE will only write non-None INCAR keys.)
        """
        basecalculator_kwargs = {key: kwargs.pop(key) for key in BaseCalculator.__init__.__annotations__ if key in kwargs}

        # BaseCalculator specific attributes
        BaseCalculator.__init__(self, **basecalculator_kwargs)

        # VASP specific attributes
        self._vasp_core: dict[str, Any] = {
            "command": command,
            "directory": directory,
            "prec": prec,
            "pp": pp,
            "kpts": kpts,
            "xc": xc,
            "encut": encut,
            "ismear": ismear,
            "sigma": sigma,
            "ediff": ediff,
            "ediffg": ediffg,
            "ibrion": ibrion,
            "isif": isif,
            "nsw": nsw,
            "nelm": nelm,
            "ispin": ispin,
            "setups": setups,
            "lreal": lreal,
            "ncore": ncore,
        }
        self._vasp_extra: dict[str, Any] = kwargs

        self._calculator: Calculator | None = None

    @property
    def calculator(self) -> Calculator:
        """Lazily construct and return the ASE `Vasp` calculator configured with stored options.

        Returns:
            Calculator: Configured `ase.calculators.vasp.Vasp` instance.
        """
        if self._calculator is None:
            from ase.calculators.vasp import Vasp

            vasp_kwargs: dict[str, Any] = {k: v for k, v in self._vasp_core.items() if v is not None}
            vasp_kwargs.update(self._vasp_extra)

            self._calculator = Vasp(**vasp_kwargs)
        return self._calculator

    def relax(
        self,
        structure: Atoms | Structure | Molecule,
        **kwargs,
    ) -> dict[str, Any]:
        """Performs relaxation on a structure.

        Args:
            structure (Atoms | Structure | Molecule | Molecule): Structure to relax.
            **kwargs: Additional keyword arguments passed to `ase.calculators.vasp.Vasp`

        Returns:
            dict: A dictionary containing the final relaxed structure and any computed properties listed in `AVAILABLE_PROPERTIES`.

            Keys in the dictionary:
                - "final_structure" (Structure): The final relaxed structure.
                - Other keys corresponding to properties in `AVAILABLE_PROPERTIES`, each containing the respective value
                  from the calculator's results.
        """
        atoms = structure.copy()

        if isinstance(atoms, (Structure, Molecule)):
            atoms = self.ase_adaptor.get_atoms(atoms)

        relax_params = {
            "isif": self._vasp_core.get("isif") or 3,
            "ibrion": self._vasp_core.get("ibrion") or 2,
            "nsw": self._vasp_core.get("nsw") or 100,
        }

        self.calculator.set(**relax_params)

        atoms.calc = self.calculator
        self.calculator.calculate(
            atoms=atoms,
            properties=self.AVAILABLE_PROPERTIES,
            system_changes=["positions", "numbers", "cell", "pbc", "initial_charges", "initial_magmoms"],
        )

        out_dict = {
            "final_structure": self.ase_adaptor.get_structure(atoms),
        }

        out_dict.update({prop: self.calculator.results[prop] for prop in self.__class__.AVAILABLE_PROPERTIES})

        return out_dict

    def calculate(
        self,
        structure: Atoms | Structure | Molecule,
    ) -> dict[str, Any]:
        """Performs a static calculation.

        Args:
            structure (Atoms | Structure | Molecule): Structure to calculate.

        Returns:
            dict: A dictionary containing the final calculated structure and any computed properties listed in
            `AVAILABLE_PROPERTIES`.
            Keys in the dictionary:
            - "final_structure" (Structure): The final calculated structure.
            - Other keys corresponding to properties in `AVAILABLE_PROPERTIES`, each containing the respective value
                  from the calculator's results.
        """
        atoms = structure.copy()

        if isinstance(atoms, (Structure, Molecule)):
            atoms = self.ase_adaptor.get_atoms(atoms)

        calculate_params = {
            "isif": None,
            "nsw": None,
            "ibrion": None,
        }

        self.calculator.set(**calculate_params)

        atoms.calc = self.calculator
        self.calculator.calculate(
            atoms=atoms,
            properties=self.AVAILABLE_PROPERTIES,
            system_changes=["positions", "numbers", "cell", "pbc", "initial_charges", "initial_magmoms"],
        )

        out_dict = {
            "final_structure": self.ase_adaptor.get_structure(atoms),
        }

        out_dict.update({prop: self.calculator.results[prop] for prop in self.__class__.AVAILABLE_PROPERTIES})

        return out_dict
