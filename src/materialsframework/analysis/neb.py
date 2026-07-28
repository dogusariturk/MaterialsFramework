"""This module provides a class to perform NEB (Nudged Elastic Band) calculations using a specified calculator.

The `NEBAnalyzer` class facilitates the calculation of minimum energy paths between two structures. It uses
`NEBTransformation` to interpolate intermediate images and optimizes the resulting path with the NEB method.
The class supports various NEB methods and allows customization of parameters such as spring constants,
climbing image, and periodic boundary conditions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ase.mep import NEB, NEBTools

from materialsframework.analysis.base import BaseAnalyzer
from materialsframework.analysis.utils import require_properties
from materialsframework.transformations.neb import NEBTransformation
from materialsframework.utils import lazy_property, to_atoms, to_structure

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class NEBAnalyzer(BaseAnalyzer):
    """A class used to perform the Nudged Elastic Band (NEB) calculation.

    Interpolates a series of images between two endpoint structures (via `NEBTransformation`) and
    optimizes them into a minimum energy path (MEP) using the NEB method.
    """

    def __init__(
        self,
        n_images: int = 5,
        spring_constant: float | list[float] = 0.1,
        climb: bool = False,
        remove_rotation_and_translation: bool = False,
        method: Literal["aseneb", "improvedtangent", "eb", "spline", "string"] = "improvedtangent",
        interpolate_lattices: bool = False,
        pbc: bool = True,
        autosort_tol: float = 0.5,
        end_amplitude: float = 1,
        calculator: BaseCalculator | None = None,
        neb_transformation: NEBTransformation | None = None,
    ) -> None:
        """Initializes the `NEBAnalyzer` object.

        Args:
            n_images (int, optional): Number of images to use in the NEB calculation. Defaults to 5.
            spring_constant (float | list[float], optional): Spring constant(s) for the NEB calculation.
                Defaults to 0.1 eV/Ang^2.
            climb (bool, optional): Whether to use the climbing image method. When `True`, the path is
                first optimized to convergence without the climbing image, then re-optimized with it
                enabled, following standard NEB practice. Defaults to False.
            remove_rotation_and_translation (bool, optional): Whether to remove rotation and translation
                of images. Defaults to False.
            method (str, optional): Method to use for the NEB calculation. Options are 'aseneb',
                'improvedtangent', 'eb', 'spline', or 'string'. Defaults to 'improvedtangent'.
            interpolate_lattices (bool, optional): Whether to interpolate lattices between images. Defaults
                to False.
            pbc (bool, optional): Whether to apply periodic boundary conditions. Defaults to True.
            autosort_tol (float, optional): Tolerance for autosorting images. Defaults to 0.5.
            end_amplitude (float, optional): Amplitude for the end images. Defaults to 1.
            calculator (BaseCalculator | None, optional): The calculator object used for energy
                calculations. Defaults to a lazily constructed default calculator.
            neb_transformation (NEBTransformation | None, optional): The transformation object used to
                generate the intermediate images. If not provided, a new instance is initialized from
                `n_images`, `interpolate_lattices`, `pbc`, `autosort_tol`, and `end_amplitude`.
        """
        super().__init__(calculator)

        # NEB specific attributes
        self.spring_constant = spring_constant
        self.climb = climb
        self.remove_rotation_and_translation = remove_rotation_and_translation
        self.method = method

        # Interpolation specific attributes
        self.n_images = n_images
        self.interpolate_lattices = interpolate_lattices
        self.pbc = pbc
        self.autosort_tol = autosort_tol
        self.end_amplitude = end_amplitude

        self.neb: NEB | None = None
        self._neb_transformation = neb_transformation

    @require_properties("energy")
    def calculate(
        self,
        initial_structure: Structure | Atoms,
        final_structure: Structure | Atoms,
        is_relaxed: bool = False,
        **kwargs: Any,
    ) -> dict[str, list[Structure] | list[float] | float | bool]:
        """Calculates the minimum energy path (MEP) between two structures.

        Interpolates intermediate images between `initial_structure` and `final_structure` (via
        `NEBTransformation`), then optimizes the resulting band with the NEB method. If the endpoints
        are not already relaxed, each is relaxed with `self.calculator` first.

        Args:
            initial_structure (Structure | Atoms): The initial structure for the NEB calculation.
            final_structure (Structure | Atoms): The final structure for the NEB calculation.
            is_relaxed (bool, optional): Whether `initial_structure` and `final_structure` are already
                relaxed. Defaults to False.
            **kwargs: Additional keyword arguments passed to the calculator's optimizer.

        Returns:
            dict[str, list[Structure] | list[float] | float | bool]: Dictionary with keys:
                - ``images``: The optimized path images as pymatgen `Structure` objects, including
                    the endpoints as the first and last elements.
                - ``energies``: Potential energy (eV) of each image after optimization.
                - ``barrier``: Forward energy barrier, from a cubic-spline fit through the images'
                    energies and forces (`ase.mep.NEBTools.get_barrier`), relative to the initial
                    (first) image.
                - ``reverse_barrier``: Reverse energy barrier, i.e. the fitted barrier minus the
                    reaction energy.
                - ``reaction_energy``: Reaction energy, i.e. the energy of the final (last) image
                    minus the energy of the initial (first) image.
                - ``converged``: Whether the last optimizer run converged within `self.calculator.steps`.

        Raises:
            ValueError: If the calculator does not implement the 'energy' property.
        """
        initial_structure = self._ensure_relaxed(initial_structure, is_relaxed)
        final_structure = self._ensure_relaxed(final_structure, is_relaxed)

        images = self.neb_transformation.apply_transformation(initial_structure, final_structure)

        atoms_images = [to_atoms(image) for image in images]
        for atoms in atoms_images:
            atoms.calc = self.calculator.calculator

        self.neb = NEB(
            images=atoms_images,
            k=self.spring_constant,
            climb=False,
            remove_rotation_and_translation=self.remove_rotation_and_translation,
            method=self.method,
            allow_shared_calculator=True,
        )

        converged = self._run_optimizer(**kwargs)

        if self.climb:
            self.neb.climb = True
            converged = self._run_optimizer(**kwargs)

        energies = [atoms.get_potential_energy() for atoms in atoms_images]
        barrier, reaction_energy = NEBTools(atoms_images).get_barrier(fit=True)

        return {
            "images": [to_structure(atoms) for atoms in atoms_images],
            "energies": energies,
            "barrier": barrier,
            "reverse_barrier": barrier - reaction_energy,
            "reaction_energy": reaction_energy,
            "converged": converged,
        }

    def _run_optimizer(self, **kwargs: Any) -> bool:
        """Runs the calculator's optimizer on `self.neb` until convergence or the step limit.

        Args:
            **kwargs: Additional keyword arguments passed to the calculator's optimizer.

        Returns:
            bool: Whether the optimizer converged within `self.calculator.steps`.
        """
        optimizer = self.calculator.optimizer(self.neb, **kwargs)
        optimizer.run(fmax=self.calculator.fmax, steps=self.calculator.steps)
        return optimizer.nsteps < self.calculator.steps

    @lazy_property("_neb_transformation")
    def neb_transformation(self) -> NEBTransformation:
        """Returns the transformation object used to generate the interpolated NEB images.

        Returns:
            NEBTransformation: The transformation object used to generate the interpolated images.
        """
        return NEBTransformation(
            n_images=self.n_images,
            interpolate_lattices=self.interpolate_lattices,
            pbc=self.pbc,
            autosort_tol=self.autosort_tol,
            end_amplitude=self.end_amplitude,
        )
