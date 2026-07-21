"""Cluster expansion module for fitting and predicting material properties.

`ClusterExpansion` builds an `icet` cluster space from a primitive structure, evaluates
training structures with a `BaseCalculator`, and fits a cross-validated model (via
`trainstation`) that predicts a chosen property, such as mixing energy, for new
configurations of the same cluster space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ase.db.sqlite import SQLite3Database

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

    from materialsframework.tools.calculator import BaseCalculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class ClusterExpansion:
    """A class to handle cluster expansion calculations."""

    def __init__(
        self,
        symprec: float = 1e-5,
        position_tolerance: float | None = None,
        is_relaxed: bool = True,
        fit_method: Literal[
            "ardr",
            "bayesian-ridge",
            "elasticnet",
            "lasso",
            "least-squares",
            "omp",
            "rfe",
            "ridge",
            "split-bregman",
        ] = "ardr",
        standardize: bool = True,
        validation_method: Literal["shuffle-split", "k-fold"] = "k-fold",
        n_splits: int = 10,
        check_condition: bool = True,
        seed: int = 42,
        verbose: bool = False,
        calculator: BaseCalculator | None = None,
    ) -> None:
        """Initialize the ClusterExpansion instance.

        Parameters:
            symprec (float): Symmetry precision for structure analysis.
            position_tolerance (float | None): Tolerance for atomic position comparison.
            is_relaxed (bool): Whether the input structures are relaxed.
            fit_method (Literal["ardr", "bayesian-ridge", "elasticnet", "lasso", "least-squares", "omp", "rfe", "ridge",
                "split-bregman"]): Method used for fitting the cluster expansion.
            standardize (bool): Whether to standardize the data before fitting.
            validation_method (Literal["shuffle-split", "k-fold"]): Method used for validation of the model.
            n_splits (int): Number of splits for cross-validation.
            check_condition (bool): Whether to check the condition number of the fit.
            seed (int): Random seed for reproducibility.
            verbose (bool): Whether to print detailed output during relaxation and fitting.
            calculator (BaseCalculator | None): Calculator for energy and property calculations.
        """
        self.symprec = symprec
        self.position_tolerance = position_tolerance
        self.is_relaxed = is_relaxed
        self.fit_method = fit_method
        self.standardize = standardize
        self.validation_method = validation_method
        self.n_splits = n_splits
        self.check_condition = check_condition
        self.seed = seed
        self.verbose = verbose

        self.cluster_space = None
        self.structure_container = None
        self.cluster_expansion = None

        self.structures = []
        self._calculator = calculator

    def fit(
        self,
        structures: list[Structure] | SQLite3Database,
        primitive_structure: Atoms,
        cutoffs: list[float],
        chemical_symbols: list[str] | list[list[str]],
        properties: list[str],
        fit_property: str = "mixing_energy",
    ):
        """Fit the cluster expansion model using the provided structures and calculator.

        Parameters:
            structures (list[Structure] | SQLite3Database): List of structures or an ASE database containing structures.
            primitive_structure (Atoms): Primitive structure for the cluster space.
            cutoffs (list[float]): Cutoff distances for the cluster space.
            chemical_symbols (list[str] | list[list[str]]): Chemical symbols for the cluster space.
            properties (list[str]): Properties to be calculated and stored in the structure container.
            fit_property (str): Property to be used for fitting the cluster expansion model.
        """
        from icet import (
            ClusterExpansion as IcetClusterExpansion,
        )
        from icet import (
            ClusterSpace,
            StructureContainer,
        )
        from trainstation import CrossValidationEstimator

        self.structures = structures

        self.cluster_space = ClusterSpace(
            structure=primitive_structure,
            cutoffs=cutoffs,
            chemical_symbols=chemical_symbols,
            symprec=self.symprec,
            position_tolerance=self.position_tolerance,
        )
        if self.verbose:
            print(self.cluster_space)

        self.structure_container = StructureContainer(cluster_space=self.cluster_space)

        if isinstance(structures, SQLite3Database):
            for row in structures.select():
                self.structure_container.add_structure(
                    structure=row.toatoms(),
                    user_tag=row.tag,
                    properties={prop: row.get(prop) for prop in properties},
                )
        else:
            for structure in self.structures:
                result = self.calculator.calculate(structure) if self.is_relaxed else self.calculator.relax(structure, verbose=self.verbose)

                self.structure_container.add_structure(
                    structure=result["final_structure"].to_ase_atoms(msonable=False),
                    properties={prop: result.get(prop) for prop in properties},
                )

        if self.verbose:
            print(self.structure_container)

        opt = CrossValidationEstimator(
            fit_data=self.structure_container.get_fit_data(key=fit_property),
            fit_method=self.fit_method,
            standardize=self.standardize,
            validation_method=self.validation_method,
            n_splits=self.n_splits,
            check_condition=self.check_condition,
            seed=self.seed,
        )
        opt.validate()
        opt.train()

        if self.verbose:
            print(opt)

        self.cluster_expansion = IcetClusterExpansion(
            cluster_space=self.cluster_space,
            parameters=opt.parameters,
            metadata=opt.summary,
        )

        if self.verbose:
            print(self.cluster_expansion)

    @property
    def calculator(self) -> BaseCalculator:
        """Returns the calculator used for energy and force calculations.

        If the calculator instance is not already initialized, this method creates a new `M3GNetCalculator` instance.

        Returns:
            BaseCalculator: The calculator object used for force and energy calculations.
        """
        if self._calculator is None:
            from materialsframework.calculators.m3gnet import M3GNetCalculator

            self._calculator = M3GNetCalculator()
        return self._calculator
