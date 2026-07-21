"""Calculator for computing potential energy, forces, stresses, and magnetic moments, and relaxing structures with the CHGNet potential."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.md import BaseMDCalculator
from materialsframework.utils import lazy_property

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class CHGNetCalculator(BaseCalculator, BaseMDCalculator):
    """Calculator for material property calculations and structure relaxation using the CHGNet potential.

    Attributes:
        AVAILABLE_PROPERTIES (list[str]): A list of properties that this calculator can compute, including "energy",
            "forces", "stress", and "magmoms".

    References:
        - CHGNet: https://doi.org/10.1038/s42256-023-00716-3
    """

    AVAILABLE_PROPERTIES = ["energy", "forces", "stress", "magmoms"]

    def __init__(
        self,
        model: str = "0.3.0",
        stress_weight: float = 1 / 160.21766208,
        include_magmoms: bool = False,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        check_cuda_mem: bool = True,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the CHGNetCalculator with the specified model and calculation settings.

        Args:
            model (str, optional): The CHGNet model to use. Defaults to "0.3.0".
            stress_weight (float, optional): Conversion factor for stress from GPa to eV/Å³. Defaults to 1 / 160.21766208.
            include_magmoms (bool, optional): Whether to include magnetic moments in the model. Defaults to False.
            on_isolated_atoms (Literal["ignore", "warn", "error"], optional): Behavior when isolated atoms are
                detected. Defaults to "warn".
            device (Literal["cpu", "cuda", "mps"], optional): The device to use for calculations. Defaults to "cpu".
            check_cuda_mem (bool, optional): Whether to check CUDA memory before running calculations. Defaults to True.
            verbose (bool, optional): Whether to print verbose output during calculations. Defaults to False.
            **kwargs: Additional keyword arguments passed to the `BaseCalculator` and `BaseMDCalculator` constructors.

        Examples:
            >>> chgnet_calculator = CHGNetCalculator(model="0.3.0", device="cuda", verbose=True)

        Note:
            The remaining parameters for the CHGNet potential are set to their default values.
        """
        super().__init__(include_magmoms=include_magmoms, **kwargs)

        # CHGNet specific attributes
        self.model = model
        self.stress_weight = stress_weight
        self.on_isolated_atoms = on_isolated_atoms
        self.device = device
        self.check_cuda_mem = check_cuda_mem
        self.verbose = verbose

        self._calculator = None

    @lazy_property("_calculator")
    def calculator(self) -> Calculator:
        """Lazily builds the ASE Calculator object for the CHGNet potential, using `device`, `check_cuda_mem`, and `stress_weight`.

        Returns:
            Calculator: The ASE Calculator object configured with the CHGNet potential.
        """
        from chgnet.model import CHGNet
        from chgnet.model import CHGNetCalculator as CHGNetASECalculator

        model = CHGNet.load(
            model_name=self.model,
            use_device=self.device,
            check_cuda_mem=self.check_cuda_mem,
            verbose=self.verbose,
        )

        return CHGNetASECalculator(
            model=model,
            use_device=self.device,
            check_cuda_mem=self.check_cuda_mem,
            stress_weight=self.stress_weight,
            on_isolated_atoms=self.on_isolated_atoms,
        )
