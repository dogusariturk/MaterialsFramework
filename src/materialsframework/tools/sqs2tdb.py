"""Python wrapper for the sqs2tdb script.

`Sqs2tdb` fits a TDB model from SQS energies using ATAT's sqs2tdb script: it copies SQS
structures from the database, calculates their energies, and fits a solution model from the results.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.atat import Mcsqs

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import default_calculator, requires

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class Sqs2tdb:
    """Python wrapper for the sqs2tdb script.

    Attributes:
        dbf (Database): The TDB database object.

    References:
        - ATAT sqs2tdb: https://doi.org/10.1016/j.calphad.2017.05.005
    """

    def __init__(
        self,
        md_temperature: float = 1000,
        md_pressure: float = 1,
        md_timestep: float = 1.0,
        md_scaling_matrix: int = 2,
        fmax: float = 0.001,
        verbose: bool = False,
        calculator: BaseCalculator | None = None,
    ) -> None:
        """Initialize the wrapper with the path to the sqs2tdb script.

        Args:
            md_temperature (float, optional): The temperature for MD calculations. Defaults to 1000 K.
            md_pressure (float, optional): The pressure for MD calculations. Defaults to 1 atm.
            md_timestep (float, optional): The timestep for MD calculations. Defaults to 1 fs.
            md_scaling_matrix (int, optional): The supercell scaling matrix applied to the 'LIQUID'
                lattice before running MD. Defaults to 2 (a 2x2x2 supercell).
            fmax (float, optional): The maximum force tolerance for relaxation. Defaults to 0.001 eV/A.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            calculator (BaseCalculator | None, optional): The calculator object used for energy calculations.
                Must also be a `BaseMDCalculator` if the 'LIQUID' lattice will be fit. Defaults to a lazily
                constructed default calculator.

        Raises:
            EnvironmentError: If the sqs2tdb script is not found in the system's PATH.
        """
        if shutil.which("sqs2tdb") is None:
            raise OSError("sqs2tdb is not installed or not found in the system's PATH.")

        self.md_temperature = md_temperature
        self.md_pressure = md_pressure
        self.md_timestep = md_timestep
        self.md_scaling_matrix = md_scaling_matrix
        self.fmax = fmax
        self.verbose = verbose

        self._calculator = calculator

        self.species = None
        self.lattices = None
        self.level = None
        self.t_min = None
        self.t_max = None
        self.sro = None
        self.bv = None
        self.phonon = None
        self.open_calphad = None
        self.terms = None

        self.dbf = None

    @requires("pycalphad", extra="calphad")
    def fit(
        self,
        species: list[str],
        lattices: list[str] | None = None,
        level: int = 1,
        t_min: float = 298.15,
        t_max: float = 10000,
        sro: bool = False,
        bv: float = 5e-3,
        phonon: bool = False,
        open_calphad: bool = False,
        terms: str | list[tuple[int, int]] | dict[str, str | list[tuple[int, int]]] | None = None,
    ) -> None:
        """Copy SQS from the database to the current directory, calculate energies, and fit a TDB model.

        Args:
            species (list): List of elements to consider (e.g., ["Al", "Ni"]).
            lattices (List[str] | None): The lattice types (e.g., ["FCC_A1", "BCC_A2"]). Defaults to all
                available lattices in the ATAT SQS database.
            level (int): The composition mesh level (e.g., 1 for midpoints). Defaults to 1.
            t_min (float): The minimum temperature for fitting. Defaults to 298.15 K.
            t_max (float): The maximum temperature for fitting. Defaults to 10000 K.
            sro (bool): Whether to include short-range order. Defaults to False.
            bv (float): The energy bump value. Defaults to 5e-3.
            phonon (bool): Whether to include phonons for end members. Defaults to False.
            open_calphad (bool): Whether to generate an Open Calphad-compliant .tdb file. Defaults to False.
            terms (str | list[tuple[int, int]] | dict[str, ...] | None): The `terms.in` content used to
                fit each lattice's solution model. Each `(order, level)` pair becomes one line of ATAT's
                `terms.in` format (order: 1 linear-in-composition, 2 binary interaction, 3 ternary, etc.;
                level: the polynomial degree of that interaction). Accepts:
                - `None` (default): use the built-in per-lattice default.
                - A single string (raw `terms.in` content) or a list of `(order, level)` pairs: applied to
                  every lattice being fit.
                - A dict keyed by lattice name (e.g. `{"BCC_A2": [(1, 0), (2, 0)]}`), each value being
                  either form above: applied per lattice, with any lattice not listed falling back to the
                  built-in default.

        Raises:
            ValueError: If the calculator object does not implement the required properties.
            ValueError: If the lattice type is not valid.
            ValueError: If "LIQUID" is among the requested lattices but the calculator does not support MD.
            ValueError: If `terms` is a dict containing a key that is not among the lattices being fit.
        """
        from pycalphad import Database

        if not all(prop in self.calculator.AVAILABLE_PROPERTIES for prop in ["energy", "forces", "stress"]):
            raise ValueError("The calculator object must have the 'energy', 'forces', and 'stress' properties implemented.")

        if lattices is None:
            lattices = self.available_lattices

        if not all(lattice in self.available_lattices for lattice in lattices):
            raise ValueError(f"Invalid lattice type. Available lattices: {self.available_lattices}")

        if "LIQUID" in lattices and not isinstance(self.calculator, BaseMDCalculator):
            raise ValueError("The calculator object must be a `BaseMDCalculator` to fit the 'LIQUID' lattice.")

        if isinstance(terms, dict) and not all(lattice in lattices for lattice in terms):
            raise ValueError(f"Invalid lattice type in `terms`. Lattices being fit: {lattices}")

        self.species = species
        self.lattices = lattices
        self.level = level
        self.t_min = t_min
        self.t_max = t_max
        self.sro = sro
        self.bv = bv
        self.phonon = phonon
        self.open_calphad = open_calphad
        self.terms = terms

        self._copy_sqs()
        self._fit_model()

        args = ["-tdb"] + (["-oc"] if open_calphad else [])
        self._run_command("sqs2tdb", args)

        tdb_filename = "_".join(sorted([s.upper() for s in self.species])) + ".tdb"
        self.dbf = Database(tdb_filename)

    @property
    def available_lattices(self) -> list[str]:
        """Get the list of available lattice types in the ATAT SQS database.

        Returns:
            List[str]: The list of available lattice types.
        """
        first_line = (Path.home() / ".atat.rc").read_text().splitlines()[0]
        base = first_line.split("=", 1)[1].strip()
        return [d.name for d in (Path(base) / "data" / "sqsdb").iterdir() if d.is_dir()]

    @property
    def calculator(self) -> BaseCalculator:
        """Returns the calculator instance used for energy, force, and stress calculations.

        If the calculator instance is not already initialized, this method returns the default calculator.

        Returns:
            BaseCalculator: The calculator object used for energy, force, and stress calculations.
        """
        if self._calculator is None:
            self._calculator = default_calculator()

        self._calculator.fmax = self.fmax
        self._calculator.verbose = self.verbose

        if isinstance(self._calculator, BaseMDCalculator):
            self._calculator.logfile = "-" if self.verbose else None
            self._calculator.temperature = self.md_temperature
            self._calculator.pressure = self.md_pressure
            self._calculator.timestep = self.md_timestep

        return self._calculator

    @staticmethod
    def _convert_structures(subdir: Path) -> None:
        """Convert the copied ATAT structure file to a POSCAR, bypassing ATAT's VASP-oriented preparation.

        Reads `str_hint.out` if present, otherwise `str.out`, exactly like ATAT's own `runstruct_vasp` does.
        Since energies are calculated with an MLIP calculator rather than real VASP, this replaces
        `runstruct_vasp -nr` (which would otherwise require a `vasp.wrap`/POTCAR/VASP installation just to
        produce a POSCAR).

        Args:
            subdir (Path): The directory containing the copied ATAT structure file.

        Raises:
            FileNotFoundError: If neither `str_hint.out` nor `str.out` exists in `subdir`.
        """
        str_file = subdir / "str_hint.out"
        if not str_file.exists():
            str_file = subdir / "str.out"
        if not str_file.exists():
            raise FileNotFoundError(f"str.out or str_hint.out does not exist in {subdir}")

        structure = Mcsqs.structure_from_str(str_file.read_text())
        structure.to(filename=str(subdir / "POSCAR"), fmt="poscar")

    def _calculate(
        self,
        subdir: Path,
        relax: bool = True,
    ) -> None:
        """Calculate SQS energies.

        This should be run inside the relevant lattice directory.

        Args:
            subdir (Path): The path to the subdirectory containing the POSCAR file.
            relax (bool): Whether to perform relaxation. Defaults to True.
        """
        structure = Structure.from_file(subdir / "POSCAR")

        if "LIQUID" in subdir.parts:
            calculator = self.calculator
            assert isinstance(calculator, BaseMDCalculator)

            structure.make_supercell(self.md_scaling_matrix)

            calculator.ensemble = "npt_nose_hoover"
            res = calculator.run(structure=structure, steps=int(3000 / self.md_timestep))  # NPT for 3 ps

            calculator.ensemble = "nvt_nose_hoover"
            res = calculator.run(structure=res["final_structure"], steps=int(10000 / self.md_timestep))  # NVT for 10 ps

            n_last = max(1, int(0.2 * len(res["total_energy"])))
            energy = np.mean(res["total_energy"][-n_last:])
            forces = np.mean(res["forces"][-n_last:], axis=0)
            stresses = np.mean(res["stresses"][-n_last:], axis=0)
            final_structure = res["final_structure"]

        else:
            res = self.calculator.relax(structure=structure) if relax else self.calculator.calculate(structure=structure)
            energy, forces, stresses, final_structure = (
                res["energy"],
                res["forces"],
                res["stress"],
                res["final_structure"],
            )

        (subdir / "energy").write_text(f"{energy:.6f}")

        final_structure.to(filename=str(subdir / "CONTCAR"), fmt="poscar")

        with (subdir / "str_relax.out").open("w") as f:
            f.write("\n".join(" ".join(map(str, row)) for row in final_structure.lattice.matrix))
            f.write("\n1 0 0\n0 1 0\n0 0 1\n")
            f.write("\n".join(" ".join(map(str, site.frac_coords)) + " " + site.species_string for site in final_structure))

        np.savetxt(str(subdir / "force.out"), forces, fmt="%.7e")

        if stresses.shape == (6,):
            from ase.stress import voigt_6_to_full_3x3_stress

            stresses = voigt_6_to_full_3x3_stress(stresses)
        np.savetxt(subdir / "stress.out", stresses, fmt="%.7e")

    def _run_command(self, command: str, args: list[str], cwd: Path | None = None) -> None:
        """Run a shell command with arguments and print stdout and stderr if verbose turned on.

        Args:
            command (str): The command to execute.
            args (list[str]): A list of arguments for the command.
            cwd (str | None): The working directory for the command.
        """
        try:
            result = subprocess.run(
                args=[command, *args],
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=60,
                check=True,
            )

            if self.verbose:
                print("STDOUT:", result.stdout.strip())
                print("STDERR:", result.stderr.strip())
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}: {e.stderr.strip()}")
        except Exception as e:
            print("Unexpected error:", e)

    def _copy_sqs(self) -> None:
        """Copy SQS from the database to the current directory and calculate energies."""
        assert self.species is not None
        assert self.lattices is not None
        species_str = ",".join(self.species)
        lattices = self.lattices

        for lattice in lattices:
            lattice_path = Path(lattice)

            for _ in range(2):
                self._run_command(
                    "sqs2tdb",
                    ["-cp", f"-l={lattice}", f"-lv={self.level}", f"-sp={species_str}"],
                )

            for wait_file in lattice_path.glob("*/wait"):
                subdir = wait_file.parent
                self._convert_structures(subdir)
                self._calculate(subdir)
                wait_file.unlink()

            if self.phonon:
                for endmember in lattice_path.glob("*/endmem"):
                    self._run_command(
                        "fitfc",
                        ["-si=str_relax.out", "-ernn=3", "-ns=1", "-nrr"],
                        cwd=endmember.parent,
                    )

                for wait_file in lattice_path.rglob("wait"):
                    subdir = wait_file.parent
                    self._convert_structures(subdir)
                    self._calculate(subdir, relax=False)
                    wait_file.unlink()

                for endmember in lattice_path.glob("*/endmem"):
                    subdir = endmember.parent
                    self._run_command("fitfc", ["-si=str_relax.out", "-f", "-frnn=1.5"], cwd=subdir)
                    self._run_command("robustrelax_vasp", ["-vib"], cwd=subdir)

    def _resolve_terms(self, lattice: str) -> str:
        """Resolve the `terms.in` content to use for a given lattice.

        Args:
            lattice (str): The lattice to resolve terms for (e.g., "FCC_A1").

        Returns:
            str: The `terms.in` file content for this lattice.
        """
        default = "1,0\n2,1" if lattice in ("BCC_A2", "FCC_A1", "HCP_A3") else "1,0:1,0\n2,0:1,0\n"
        terms = self.terms

        if terms is None:
            spec = default
        elif isinstance(terms, dict):
            spec = terms.get(lattice, default)
        else:
            spec = terms

        if isinstance(spec, str):
            return spec
        return "\n".join(f"{order},{level}" for order, level in spec)

    def _fit_model(self) -> None:
        """Fit a solution model from SQS energies.

        This should be run inside the relevant lattice directory.
        """
        assert self.lattices is not None
        for lattice in self.lattices:
            args = ["-fit", f"-Tl={self.t_min}", f"-Tu={self.t_max}"]
            if self.bv:
                args.append(f"-bv={self.bv}")
            if self.sro:
                args.append("-sro")

            lattice_path = Path(lattice).resolve()
            self._run_command("sqs2tdb", args, cwd=lattice_path)

            (lattice_path / "terms.in").write_text(self._resolve_terms(lattice))

            self._run_command("sqs2tdb", args, cwd=lattice_path)
