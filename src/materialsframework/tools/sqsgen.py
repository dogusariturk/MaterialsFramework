"""This module provides a class for generating Special Quasirandom Structures (SQS).

The `SqsGenerator` class generates SQS structures that mimic the statistical properties of a
random alloy, using the method implemented in `sqsgenerator`. SQS structures approximate
randomness while keeping simulations of disordered systems and alloys computationally tractable.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pymatgen.core import Composition, Lattice

from materialsframework.utils import requires

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class SqsGenerator:
    """A class used to generate Special Quasirandom Structures (SQS).

    Generates structures that approximate a random arrangement of atoms while matching specific
    pair and multi-site correlation functions.
    """

    def __init__(
        self,
        iterations: int = 1000,
        make_structures: bool = True,
        mode: Literal["random", "systematic"] = "random",
        structure_format: str = "pymatgen",
        log_level: Literal["trace", "debug", "info", "warning", "error"] = "warning",
    ) -> None:
        """Initializes the `SqsGenerator` object.

        Args:
            iterations (int, optional): The number of iterations for the SQS generation. Defaults to 1000.
            make_structures (bool, optional): Whether to generate the structures during the optimization process. Defaults to True.
            mode (Literal["random", "systematic"], optional): The mode for SQS generation. Defaults to "random".
            structure_format (str, optional): The structure format for the generated SQS structure. Defaults to "pymatgen".
            log_level (Literal["trace", "debug", "info", "warning", "error"], optional): The log level for the SQS
                generation. Defaults to "warning".
        """
        self._iterations = iterations
        self._make_structures = make_structures
        self._mode = mode
        self._structure_format = structure_format
        self._log_level = log_level

    @requires("sqsgenerator", extra="sqsgen")
    def generate(
        self,
        composition: Composition | str,
        crystal_structure: str = "FCC",
        supercell_size: tuple[int, int, int] = (5, 5, 5),
        shell_weights: dict[int, float] | None = None,
    ) -> dict[str, Any]:
        """Generates a supercell using the SQS (Special Quasirandom Structures) method.

        Args:
            composition (Composition | str): The composition of the supercell.
            crystal_structure (str, optional): The crystal structure of the supercell. Defaults to "FCC".
            supercell_size (tuple[int, int, int], optional): The size of the supercell. Defaults to (5, 5, 5).
            shell_weights (dict[int, float], optional): The weights for the coordination shells. Defaults to {1: 1.0, 2: 0.5}.

        Returns:
            dict[str, Any]: Dictionary with keys:
                - ``structure``: Generated SQS structure.
                - ``objective``: Final objective value from SQS optimization.

        Raises:
            ValueError: If the crystal structure is invalid.
        """
        from sqsgenerator import optimize, parse_config
        from sqsgenerator.core import LogLevel

        if isinstance(composition, str):
            composition = Composition(composition)

        lattice = self._get_lattice(composition=composition, crystal_structure=crystal_structure.lower())
        coords = self._get_coords(crystal_structure=crystal_structure.lower())
        multiplier = self._get_multiplier(crystal_structure=crystal_structure.lower())
        sqs_composition = self._determine_composition(
            supercell_size=supercell_size, composition=composition, multiplier=multiplier
        )

        if shell_weights is None:
            shell_weights = {1: 1.0} if supercell_size == (1, 1, 1) else {1: 1.0, 2: 0.5}

        configuration = {
            "structure": {
                "lattice": lattice.matrix,
                "coords": coords,
                "species": ["W"] * multiplier,  # Tungsten used here as a placeholder element
                "supercell": supercell_size,
            },
            "iterations": self._iterations,
            "shell_weights": shell_weights,
            "composition": sqs_composition,
            "iteration_mode": self._mode,
        }

        _log_level_map = {
            "trace": LogLevel.trace,
            "debug": LogLevel.debug,
            "info": LogLevel.info,
            "warning": LogLevel.warn,
            "error": LogLevel.error,
        }

        results = optimize(
            parse_config(configuration),
            level=_log_level_map.get(self._log_level, LogLevel.warn),
        )

        sqs = self._parse_results_for_structure(results)
        objective = self._parse_results_for_objective(results)

        return {"structure": sqs, "objective": objective}

    @staticmethod
    def _get_lattice(composition: Composition, crystal_structure: str) -> Lattice:
        """Calculates and returns the lattice for the given composition and crystal structure.

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
                a=avg_radius * 2, c=avg_radius * 2 * np.sqrt(8.0 / 3.0)
            ).get_niggli_reduced_lattice(),
            "dhcp": lambda: Lattice.hexagonal(
                a=avg_radius * 2, c=avg_radius * 2 * np.sqrt(8.0 / 3.0) * 2
            ).get_niggli_reduced_lattice(),
            "fcc_prim": lambda: Lattice(
                matrix=[
                    [0, avg_radius * np.sqrt(2), avg_radius * np.sqrt(2)],
                    [avg_radius * np.sqrt(2), 0, avg_radius * np.sqrt(2)],
                    [avg_radius * np.sqrt(2), avg_radius * np.sqrt(2), 0],
                ]
            ),
            "fcc": lambda: Lattice.cubic(a=avg_radius * 2 * np.sqrt(2)),
            "bcc": lambda: Lattice.cubic(a=avg_radius * 4 / np.sqrt(3)),
            "b2": lambda: Lattice.cubic(a=avg_radius * 4 / np.sqrt(3)),
            "sc": lambda: Lattice.cubic(a=avg_radius),
        }

        if crystal_structure not in lattice_creators:
            raise ValueError(f"Invalid crystal structure: {crystal_structure!r}")

        return lattice_creators[crystal_structure]()

    @staticmethod
    def _get_coords(crystal_structure) -> list[list[float]]:
        """Returns the coordinates of atoms based on the crystal structure.

        Args:
            crystal_structure (str): The crystal structure of the supercell.

        Returns:
            list[list[float]]: The coordinates of atoms based on the crystal structure.

        Raises:
            ValueError: If the crystal structure is invalid.
        """
        coords_creators = {
            "hcp": [
                [1.0 / 3.0, 2.0 / 3.0, 1.0 / 4.0],
                [2.0 / 3.0, 1.0 / 3.0, 3.0 / 4.0],
            ],
            "dhcp": [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0 / 2.0],
                [1.0 / 3.0, 2.0 / 3.0, 1.0 / 4.0],
                [2.0 / 3.0, 1.0 / 3.0, 3.0 / 4.0],
            ],
            "fcc_prim": [[0.0, 0.0, 0.0]],
            "fcc": [[0.0, 0.0, 0.0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0.0, 0.5, 0.5]],
            "bcc": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            "b2": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            "sc": [[0.0, 0.0, 0.0]],
        }

        if crystal_structure not in coords_creators:
            raise ValueError(f"Invalid crystal structure: {crystal_structure!r}")

        return coords_creators[crystal_structure]

    @staticmethod
    def _get_multiplier(crystal_structure) -> int:
        """Returns the multiplier for the given crystal structure.

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

        if crystal_structure not in multiplier_creators:
            raise ValueError(f"Invalid crystal structure: {crystal_structure!r}")

        return multiplier_creators[crystal_structure]

    def _determine_composition(
        self, supercell_size: tuple[int, int, int], composition: Composition, multiplier: int
    ) -> dict[str, int]:
        """Determines the composition of the supercell.

        Rounds each element's fractional share to the nearest atom count, then assigns any
        leftover atoms (from independent rounding not summing exactly) to the largest-share
        element, so the counts always add up to the exact number of sites in the supercell.

        Args:
            supercell_size (tuple[int, int, int]): The size of the supercell.
            composition (Composition): The composition of the supercell.
            multiplier (int): The multiplier for the crystal structure.

        Returns:
            dict[str, int]: A dictionary containing the element symbols as keys and the corresponding
            number of atoms as values.
        """
        result = multiplier * reduce(operator.mul, supercell_size)

        fractions = composition.fractional_composition.as_reduced_dict()
        counts = {el: round(amt * result) for el, amt in fractions.items()}

        shortfall = result - sum(counts.values())
        if shortfall:
            largest = max(fractions, key=fractions.get)
            counts[largest] += shortfall

        return counts

    @staticmethod
    @requires("sqsgenerator", extra="sqsgen")
    def _parse_results_for_structure(results: Any) -> Structure:
        """Parses the results dictionary from the generate function to extract the SQS structure.

        Args:
            results (Any): The results object returned by the SQS optimization.

        Returns:
            Structure: The SQS structure generated by the calculator.
        """
        from sqsgenerator import to_pymatgen

        return to_pymatgen(results.best().structure()).get_sorted_structure()

    @staticmethod
    def _parse_results_for_objective(results: Any) -> float:
        """Parses the results dictionary from the generate function to extract the objective value.

        Args:
            results (Any): The results object returned by the SQS optimization.

        Returns:
            float: The objective value of the SQS structure.
        """
        return results.best().objective
