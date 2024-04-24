"""
This module provides classes to perform calculations using the CHGNet potential.
"""
from typing import Dict, Optional, Tuple, Union

from chgnet.model import CHGNet, CHGNetCalculator as CHGNetCalc, StructOptimizer
from numpy import ndarray
from pymatgen.core import Structure

from materialsframework.calculators import Calculator, Relaxer

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class CHGNetRelaxer(Relaxer):
    """
    A class used to represent a CHGNet Relaxer.

    This class provides methods to perform relaxation of a structure using the CHGNet potential.
    """

    def __init__(
            self,
            model: str = "0.3.0",
            use_device: Optional[str] = None,
            fmax: Optional[float] = 0.1,
            steps: Optional[int] = 500,
            relax_cell: Optional[bool] = True,
            ase_filter: Optional[str] = "FrechetCellFilter",
            assign_magmoms: bool = True,
            verbose: bool = True
    ) -> None:
        """
        Initializes the CHGNet calculator.

        Args:
            model (str): The CHGNet model to use. Defaults to "0.3.0".
            use_device (Optional[str]): The device to use for calculations. Defaults to None.
            fmax (Optional[float]): The maximum force tolerance for convergence. Defaults to 0.1.
            steps (Optional[int]): The maximum number of optimization steps. Defaults to 500.
            relax_cell (Optional[bool]): Whether to relax the lattice cell. Defaults to True.
            ase_filter (Optional[str]): The ASE filter to use for relaxation. Defaults to "FrechetCellFilter".
            assign_magmoms (bool): Whether to assign magnetic moments to the atoms. Defaults to True.
            verbose (bool): Whether to print verbose output during calculations. Defaults to True.

        Examples:
            >>> relaxer = CHGNetRelaxer()
            >>> relaxer = CHGNetRelaxer(model="0.3.0", use_device="cuda", fmax=0.1, steps=500,
            ...                         relax_cell=True, ase_filter="FrechetCellFilter", assign_magmoms=True,
            ...                         verbose=True)

        Note:
            The remaining values for the arguments are set to the default values for the CHGNet potential.
        """
        self._model = model
        self._use_device = use_device
        self._fmax = fmax
        self._steps = steps
        self._relax_cell = relax_cell
        self._ase_filter = ase_filter
        self._assign_magmoms = assign_magmoms
        self._verbose = verbose

        self._relaxer = None
        self._potential = None

    @property
    def potential(self) -> CHGNet:
        """
        Returns the CHGNet potential associated with this instance.

        If the potential has not been initialized yet, it will be loaded
        using the model attribute of this instance.

        Returns:
            CHGNet: The CHGNet potential associated with this instance.
        """
        if self._potential is None:
            self._potential = CHGNet.load(model_name=self._model,
                                          use_device=self._use_device,
                                          verbose=self._verbose)
        return self._potential

    @property
    def relaxer(self) -> StructOptimizer:
        """
        Returns the Relaxer object associated with this instance.

        If the Relaxer object has not been initialized yet, it will be created using the
        potential and relax_cell attributes of this instance.

        Returns:
            StructOptimizer: The Relaxer object associated with this instance.
        """
        if self._relaxer is None:
            self._relaxer = StructOptimizer(model=self.potential,
                                            use_device=self._use_device)
        return self._relaxer

    def relax(self, structure: Structure) -> Tuple[Structure, float]:
        """
        Performs the relaxation of the structure using the CHGNet calculator.

        Args:
            structure (Structure): The input structure.

        Returns:
            Tuple[Structure, float]: A tuple containing the relaxed structure and its energy.

        Examples:
            >>> relaxer = CHGNetRelaxer()
            >>> struct = Structure.from_file("POSCAR")
            >>> relaxation_results = relaxer.relax(structure=struct)
        """
        relax_results = self.relaxer.relax(atoms=structure,
                                           fmax=self._fmax,
                                           steps=self._steps,
                                           relax_cell=self._relax_cell,
                                           ase_filter=self._ase_filter,
                                           verbose=self._verbose,
                                           assign_magmoms=self._assign_magmoms)
        return {
                'final_structure': relax_results["final_structure"],
                'energy': float(relax_results["trajectory"].energies[-1]),
                'magmom': relax_results["final_structure"].site_properties["magmom"]
        }


class CHGNetCalculator(Calculator):
    """
    A class used to represent a CHGNet Calculator.

    This class is used to calculate the potential energy, forces, stresses,
    and magmoms of a given structure using the CHGNet potential.
    """

    def __init__(
            self,
            model: str = "0.3.0",
            use_device: Optional[str] = None,
            verbose: bool = True
    ) -> None:
        """
        Initializes the CHGNet calculator.

        Args:
            model (str): The CHGNet model to use.
            use_device (Optional[str]): The device to use for calculations. Defaults to None.
            verbose (bool): Whether to print verbose output during calculations. Defaults to True.

        Examples:
            >>> calculator = CHGNetCalculator()
            >>> calculator = CHGNetCalculator(model="0.3.0", use_device="cuda", verbose=True)

        Note:
            The remaining values for the arguments are set to the default values for the CHGNet potential.
        """
        self._model: str = model
        self._use_device = use_device
        self._verbose = verbose

        self._calculator = None
        self._potential = None

    @property
    def potential(self) -> CHGNet:
        """
        Returns the CHGNet potential associated with this instance.

        If the potential has not been initialized yet, it will be loaded
        using the model attribute of this instance.

        Returns:
            CHGNet: The CHGNet potential associated with this instance.
        """
        if self._potential is None:
            self._potential = CHGNet.load(model_name=self._model,
                                          use_device=self._use_device,
                                          verbose=self._verbose)
        return self._potential

    @property
    def calculator(self) -> CHGNetCalc:
        """
        Returns the CHGNet CHGNetCalculator instance.

        If the calculator instance is not already created, it creates a new CHGNetCalc instance
        with the specified potential and returns it. Otherwise, it returns the existing
        calculator instance.

        Returns:
            CHGNetCalc: The CHGNet calculator instance.
        """
        if self._calculator is None:
            self._calculator = CHGNetCalc(model=self.potential,
                                          use_device=self._use_device,
                                          verbose=self._verbose)
        return self._calculator

    def calculate(
            self,
            structure: Structure
    ) -> Dict[str, Union[float, ndarray]]:
        """
        Calculates the potential energy, forces, stresses, and magmoms of the given structure.

        Args:
            structure (Structure): The structure for which the properties will be calculated.

        Returns:
            Dict[str, Any]: A dictionary containing the calculated properties.

        Examples:
            >>> calculator = CHGNetCalculator()
            >>> struct = Structure.from_file("POSCAR")
            >>> calculation_results = calculator.calculate(structure=struct)
        """
        atoms = structure.to_ase_atoms()
        self.calculator.calculate(atoms=atoms)
        return {
                "potential_energy": self.calculator.results["energy"],
                "forces": self.calculator.results["forces"],
                "stress": self.calculator.results['stress'],
                "magmoms": self.calculator.results['magmoms']
        }
