"""
This module provides a class to perform NEB (Nudged Elastic Band) calculations using a specified calculator.

The `NEBAnalyzer` class facilitates the calculation of minimum energy paths between two structures. It uses
interpolation to generate intermediate images and optimizes the path using the NEB method. The class supports
various NEB methods and allows customization of parameters such as spring constants, climbing image, and
periodic boundary conditions.
"""
from __future__ import annotations

from typing import Literal

from ase.mep import NEB
from pymatgen.core import Structure

from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class NEBAnalyzer:
    """
    A class used to perform the Nudged Elastic Band (NEB) calculation.
    """

    def __init__(
            self,
            k: float | list[float] = 0.1,
            climb: bool = False,
            remove_rotation_and_translation: bool = False,
            method: Literal['aseneb', 'improvedtangent', 'eb', 'spline', 'string'] = 'aseneb',
            n_images: int = 5,
            interpolate_lattices: bool = False,
            pbc: bool = True,
            autosort_tol: float = 0.5,
            end_amplitude: float = 1,
            calculator: BaseCalculator | None = None
    ) -> None:
        """
        Initialize the NEB class.

        Args:
            k (float | list[float], optional): Spring constant(s) for the NEB calculation. Defaults to 0.1 eV/Ang.
            climb (bool, optional): Whether to use the climbing image method. Defaults to False.
            remove_rotation_and_translation (bool, optional): Whether to remove rotation and translation of images.
                                                            Defaults to False.
            method (str, optional): Method to use for NEB calculation. Options are 'aseneb', 'improvedtangent', 'eb',
                                      'spline', or 'string'. Defaults to 'aseneb'.
            n_images (int, optional): Number of images to use in the NEB calculation. Defaults to 5.
            interpolate_lattices (bool, optional): Whether to interpolate lattices between images. Defaults to False.
            pbc (bool, optional): Whether to apply periodic boundary conditions. Defaults to True.
            autosort_tol (float, optional): Tolerance for autosorting images. Defaults to 0.5.
            end_amplitude (float, optional): Amplitude for the end images. Defaults to 1.
            calculator (BaseCalculator | None, optional): The calculator object used for energy calculations.
                                                          Defaults to `M3GNetCalculator`.
        """
        # NEB specific attributes
        self.k = k
        self.climb = climb
        self.remove_rotation_and_translation = remove_rotation_and_translation
        self.method = method

        # Interpolation specific attributes
        self.n_images = n_images
        self.interpolate_lattices = interpolate_lattices
        self.pbc = pbc
        self.autosort_tol = autosort_tol
        self.end_amplitude = end_amplitude

        self.neb = None
        self._calculator = calculator

    def calculate(
            self,
            initial_structure: Structure,
            final_structure: Structure,
            is_relaxed: bool = False,
            **kwargs
    ) -> None:
        """
        Perform the NEB calculation between two structures.

        This method generates intermediate images between the initial and final structures, applies the NEB method,
        and optimizes the path using the specified calculator. If the structures are not relaxed, they will be relaxed
        before the NEB calculation.

        Args:
            initial_structure (Structure): The initial structure for the NEB calculation.
            final_structure (Structure): The final structure for the NEB calculation.
            is_relaxed (bool, optional): Whether the structures are already relaxed. Defaults to False.
            **kwargs: Additional keyword arguments passed to the calculator's optimizer.

        Raises:
            ValueError: If the calculator does not implement the 'energy' property.
        """
        if "energy" not in self.calculator.AVAILABLE_PROPERTIES:
            raise ValueError("The calculator object must have the 'energy' property implemented.")

        if not is_relaxed:
            initial_structure: Structure = self.calculator.relax(initial_structure)["final_structure"]
            final_structure: Structure = self.calculator.relax(final_structure)["final_structure"]

        images = initial_structure.interpolate(
                end_structure=final_structure,
                nimages=self.n_images,
                interpolate_lattices=self.interpolate_lattices,
                pbc=self.pbc,
                autosort_tol=self.autosort_tol,
                end_amplitude=self.end_amplitude,
        )

        images = [image.to_ase_atoms(msonable=False) for image in images]
        for image in images:
            image.calc = self.calculator.calculator

        self.neb = NEB(
                images=images,
                k=self.k,
                climb=self.climb,
                remove_rotation_and_translation=self.remove_rotation_and_translation,
                method=self.method,
                allow_shared_calculator=True
        )

        optimizer = self.calculator.optimizer(self.neb, **kwargs)
        optimizer.run(fmax=self.calculator.fmax)

    @property
    def calculator(self) -> BaseCalculator:
        """
        Returns the calculator instance used for energy, force, and stress calculations.

        If the calculator instance is not already initialized, this method creates a new `M3GNetCalculator` instance.

        Returns:
            BaseCalculator: The calculator object used for energy, force, and stress calculations.
        """
        if self._calculator is None:
            from materialsframework.calculators.m3gnet import M3GNetCalculator
            self._calculator = M3GNetCalculator()
        return self._calculator
