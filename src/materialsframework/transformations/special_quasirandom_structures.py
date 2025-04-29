"""
This module contains classes for generating Special Quasirandom Structures (SQS).

The `SqsgenTransformation` class allows users to generate SQS structures, which are designed
to mimic the statistical properties of a random alloy, using the method implemented in the `sqsgenerator`.
SQS structures are important in simulating disordered systems or alloys, as they approximate randomness
while maintaining computational tractability.
"""
from __future__ import annotations

import operator
from functools import reduce
from typing import Any, Literal, TYPE_CHECKING

import numpy as np
from pymatgen.core import Lattice
from sqsgenerator import sqs_optimize

if TYPE_CHECKING:
    from pymatgen.core import Composition, Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class SqsgenTransformation:
    """
    A class used to generate Special Quasirandom Structures (SQS).

    The `SqsgenTransformation` class provides methods to generate SQS structures using the SQS method,
    which produces structures that approximate a random arrangement of atoms while matching specific
    pair and multi-site correlation functions. These structures are widely used in simulations of
    disordered systems and alloys.
    """

    def __init__(
            self,
            iterations: int = 1000,
            make_structures: bool = True,
            mode: Literal["random", "systematic"] = "random",
            structure_format: str = "pymatgen",
            log_level: Literal["trace", "debug", "info", "warning", "error"] = "warning"
    ) -> None:
        """
        Initializes the `SqsgenTransformation` object.

        Args:
            iterations (int, optional): The number of iterations for the SQS generation. Defaults to 1000.
            make_structures (bool, optional): Whether to generate the structures during the optimization process. Defaults to True.
            mode (Literal["random", "systematic"], optional): The mode for SQS generation. Defaults to "random".
            structure_format (str, optional): The structure format for the generated SQS structure. Defaults to "pymatgen".
            log_level (Literal["trace", "debug", "info", "warning", "error"], optional): The log level for the SQS generation. Defaults to "warning".
        """
        self._iterations = iterations
        self._make_structures = make_structures
        self._mode = mode
        self._structure_format = structure_format
        self._log_level = log_level

        self._lattice: Lattice | None = None
        self._coords: dict[str, list[float]] | None = None
        self._multiplier: int | None = None
        self._supercell_size: tuple[int, int, int] | None = None
        self._composition: dict[str, int] | None = None

        self._sqs: Structure | None = None
        self._objective: float | None = None

        self.results: dict[int, dict[str, Any]] | None = None
        self.timings: dict[int, float | list[float]] | None = None

    def generate(
            self,
            composition: Composition,
            crystal_structure: str = "FCC",
            supercell_size: tuple[int, int, int] = (5, 5, 5),
            shell_weights: dict[int, float] | None = None,
    ) -> dict[str, Any]:
        """
        Generates a supercell using the SQS (Special Quasirandom Structures) method.

        Args:
            composition (Composition): The composition of the supercell.
            crystal_structure (str, optional): The crystal structure of the supercell. Defaults to "FCC".
            supercell_size (tuple[int, int, int], optional): The size of the supercell. Defaults to (5, 5, 5).
            shell_weights (dict[int, float], optional): The weights for the coordination shells. Defaults to {1: 1.0, 2: 0.5}.

        Returns:
            dict[str, Any]: A dictionary with the following keys:
                - "structure" (Any): The generated SQS structure.
                - "objective" (float): The final objective value from the SQS optimization.

        Raises:
            ValueError: If the crystal structure is invalid.
        """
        self._supercell_size = supercell_size
        self._lattice = self._get_lattice(composition=composition,
                                          crystal_structure=crystal_structure.lower())
        self._coords = self._get_coords(crystal_structure=crystal_structure.lower())
        self._multiplier = self._get_multiplier(crystal_structure=crystal_structure.lower())
        self._composition = self._determine_composition(supercell_size=self._supercell_size,
                                                        composition=composition)

        shell_weights = {1: 1.0, 2: 0.5} if shell_weights is None else {1: 1.0} if supercell_size == (1, 1, 1) else shell_weights

        configuration = {
                "structure": {
                        "lattice": self._lattice.matrix,
                        "coords": self._coords,
                        "species": ["W"] * self._multiplier,  # Tungsten used here as a placeholder element
                        "supercell": self._supercell_size,
                },
                "iterations": self._iterations,
                "shell_weights": shell_weights,
                "composition": self._composition,
                "mode": self._mode,
        }

        self.results, self.timings = sqs_optimize(
                settings=configuration,
                make_structures=self._make_structures,
                structure_format=self._structure_format,
                log_level=self._log_level
        )

        self._sqs = self._parse_results_for_structure()
        self._objective = self._parse_results_for_objective()

        return {
                "structure": self._sqs,
                "objective": self._objective
        }

    @staticmethod
    def _get_lattice(
            composition: Composition,
            crystal_structure: str
    ) -> Lattice:
        """
        Calculates and returns the lattice for the given composition and crystal structure.

        BE CAREFUL: This function returns primitive unit cells for the HCP and DHCP structures by default.

        Args:
            composition (Composition): The composition of the supercell.
            crystal_structure (str): The crystal structure of the supercell.

        Returns:
            Lattice: The calculated lattice.
        """

        avg_radius = np.sum([el.atomic_radius * amt for (el, amt) in composition.fractional_composition.items()])

        lattice_creators = {
                "hcp": lambda: Lattice.hexagonal(
                        a=avg_radius * 2,
                        c=avg_radius * 2 * np.sqrt(8.0 / 3.0)).get_niggli_reduced_lattice(),
                "dhcp": lambda: Lattice.hexagonal(
                        a=avg_radius * 2,
                        c=avg_radius * 2 * np.sqrt(8.0 / 3.0) * 2).get_niggli_reduced_lattice(),
                "fcc_prim": lambda: Lattice(
                        matrix=[[0, avg_radius * np.sqrt(2), avg_radius * np.sqrt(2)],
                                [avg_radius * np.sqrt(2), 0, avg_radius * np.sqrt(2)],
                                [avg_radius * np.sqrt(2), avg_radius * np.sqrt(2), 0]]),
                "fcc": lambda: Lattice.cubic(a=avg_radius * 2 * np.sqrt(2)),
                "bcc": lambda: Lattice.cubic(a=avg_radius * 4 / np.sqrt(3)),
                "b2": lambda: Lattice.cubic(a=avg_radius * 4 / np.sqrt(3)),
                "sc": lambda: Lattice.cubic(a=avg_radius)
        }

        return lattice_creators.get(crystal_structure, lambda: ValueError("Invalid crystal structure."))()

    @staticmethod
    def _get_coords(crystal_structure) -> dict[str, list[float]]:
        """
        Returns the coordinates of atoms based on the crystal structure.

        Args:
            crystal_structure (str): The crystal structure of the supercell.

        Returns:
            dict[str, list[float]: The coordinates of atoms based on the crystal structure.

        Raises:
            ValueError: If the crystal structure is invalid.
        """
        coords_creators = {
                "hcp": [[1.0 / 3.0, 2.0 / 3.0, 1.0 / 4.0],
                        [2.0 / 3.0, 1.0 / 3.0, 3.0 / 4.0]],
                "dhcp": [[0, 0, 0],
                         [0, 0, 1.0 / 2.0],
                         [1.0 / 3.0, 2.0 / 3.0, 1.0 / 4.0],
                         [2.0 / 3.0, 1.0 / 3.0, 3.0 / 4.0], ],
                "fcc_prim": [[0.0, 0.0, 0.0]],
                "fcc": [[0.0, 0.0, 0.0],
                        [0.5, 0.5, 0],
                        [0.5, 0, 0.5],
                        [0.0, 0.5, 0.5]],
                "bcc": [[0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.5]],
                "b2": [[0.0, 0.0, 0.0],
                       [0.5, 0.5, 0.5]],
                "sc": [[0.0, 0.0, 0.0]],
        }

        return coords_creators.get(crystal_structure, ValueError("Invalid crystal structure."))

    @staticmethod
    def _get_multiplier(crystal_structure) -> int:
        """
        Returns the multiplier for the given crystal structure.

        Args:
            crystal_structure (str): The crystal structure of the supercell.

        Returns:
            int: The multiplier for the given crystal structure.

        Raises:
            ValueError: If the crystal structure is invalid.
        """
        multiplier_creators = {
                "hcp": 2,
                "dhcp": 4,
                "fcc_prim": 1,
                "fcc": 4,
                "bcc": 2,
                "b2": 2,
                "sc": 1,
        }

        return multiplier_creators.get(crystal_structure, ValueError("Invalid crystal structure."))

    def _determine_composition(
            self,
            supercell_size: tuple[int, int, int],
            composition: Composition
    ) -> dict[str, int]:
        """
        Determines the composition of the supercell.

        Args:
            supercell_size (tuple[int, int, int]): The size of the supercell.
            composition (Composition): The composition of the supercell.

        Returns:
            dict[str, int]: A dictionary containing the element symbols as keys and the corresponding
            number of atoms as values.
        """
        result = self._multiplier * reduce(operator.mul, supercell_size)

        return {
                el: int(round(amt, 5) * result)
                for el, amt in composition.fractional_composition.to_reduced_dict.items()
        }

    def _parse_results_for_structure(self) -> Structure:
        """
        Parses the results dictionary from the generate function.

        This function takes the results dictionary generated by the generate function and
        parses it to extract the SQS structure.

        Returns:
            Structure: The SQS structure generated by the calculator.
        """
        return next(iter(self.results.values()))["structure"].sort()

    def _parse_results_for_objective(self) -> float:
        """
        Parses the results dictionary from the generate function.

        This function takes the results dictionary generated by the generate function and
        parses it to extract the objective value.

        Returns:
            float: The objective value of the SQS structure.
        """
        return next(iter(self.results.values()))["objective"]

    @property
    def sqs(self) -> Structure:
        """
        Returns the SQS generated by the calculator.

        Raises:
            ValueError: If the SQS has not been generated yet.

        Returns:
            Structure: The SQS structure.
        """
        if self._sqs is None:
            raise ValueError("SQS has not been generated yet.")

        return self._sqs

    @property
    def objective(self) -> float:
        """
        Returns the objective value of the SQS generated by the calculator.

        Raises:
            ValueError: If the SQS has not been generated yet.

        Returns:
            float: The objective value of the SQS structure.
        """
        if self._objective is None:
            raise ValueError("SQS has not been generated yet.")

        return self._objective
