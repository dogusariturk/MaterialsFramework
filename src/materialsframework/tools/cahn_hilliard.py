"""This module implements a Cahn-Hilliard phase field model for simulating phase separation.

It uses a finite difference method to solve the Cahn-Hilliard equation and includes functionality for
visualizing the results.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit

from materialsframework.utils import requires

if TYPE_CHECKING:
    from pycalphad import Database

__authors__ = ["Doguhan Sariturk", "Vahid Attari"]
__maintainer__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


class SimulationGrid:
    """Handles the simulation grid and phase field variables."""

    def __init__(
        self,
        nx: int = 128,
        ny: int = 128,
        lx: float = 2e-6,
        ly: float = 2e-6,
        dt: float = 1e-12,
    ) -> None:
        """Initializes the simulation grid with given parameters.

        Args:
            nx (int, optional): Number of grid points in x-direction. Defaults to 128.
            ny (int, optional): Number of grid points in y-direction. Defaults to 128.
            lx (float, optional): Length of the grid in x-direction. Defaults to 2e-6.
            ly (float, optional): Length of the grid in y-direction. Defaults to 2e-6.
            dt (float, optional): Time step for the simulation. Defaults to 1e-12.
        """
        self.nx, self.ny = nx, ny
        self.lx, self.ly = lx, ly
        self.dx = self.lx / (self.nx - 1)
        self.dt = dt

        # Precompute constant for the 9-point Laplacian stencil
        self.laplace_factor = 1.0 / (4.0 * self.dx**2)

        self.phi = np.zeros((nx, ny), dtype=np.float64)
        self.lap_phi = np.zeros((nx, ny), dtype=np.float64)
        self.chem_pot = np.zeros((nx, ny), dtype=np.float64)
        self.lap_chem_pot = np.zeros((nx, ny), dtype=np.float64)


class MaterialParameters:
    """Stores material parameters such as energy and kinetic properties."""

    @requires("pycalphad", extra="calphad")
    def __init__(
        self,
        db: Database | str,
        temperature: float,
        component: str,
        composition: float,
        elements: list[str] | None = None,
        phase: str | None = None,
    ) -> None:
        """Initializes material parameters with given composition and potential values.

        Args:
            db (Database | str): pycalphad Database object or path to the database file.
            temperature (float): Temperature in Kelvin.
            component (str): Component name.
            composition (float): Composition value.
            elements (list[str] | None, optional): List of elements. Defaults to None.
            phase (str | None, optional): Phase name. Defaults to None.

        Raises:
            ValueError: If multiple phases are found in the database and no phase is specified.
        """

        def energy(x, a, b, c, d, e, f, g, h, i, j, k):
            """Polynomial function for fitting."""
            return a * x**10 + b * x**9 + c * x**8 + d * x**7 + e * x**6 + f * x**5 + g * x**4 + h * x**3 + i * x**2 + j * x + k

        from pycalphad import Database, calculate

        dbf = db if isinstance(db, Database) else Database(db)

        if phase is None and len(dbf.phases) > 1:
            raise ValueError("Multiple phases found in the database. Please specify a phase.")

        comps = sorted(dbf.elements) if elements is None else elements
        if "/-" in comps:
            comps.remove("/-")
        phases = [phase] if phase is not None else list(dbf.phases)

        gs = calculate(dbf, comps, phases, T=temperature)
        xs = gs.X.sel(component=component.upper()).values.ravel()
        ys = gs.GM.values.ravel()
        popt, _ = curve_fit(f=energy, xdata=xs, ydata=ys)

        self.coeffs = np.array(popt, dtype=np.longdouble)
        self.composition = composition
        self.mobility = np.longdouble(1.0e-10)
        self.kappa = np.longdouble(1e-11)

        self.free_energy_poly_deriv = np.poly1d(self.coeffs).deriv()


class PhaseFieldModel:
    """Implements the Cahn-Hilliard solver with output visualization."""

    def __init__(
        self,
        material_properties: MaterialParameters,
        simulation_grid: SimulationGrid | None = None,
        wrt_cycle: int = 5000,
        stop_iter: int = 50000,
        seed: int = 42,
    ) -> None:
        """Initializes the phase field model with simulation grid and material properties.

        Args:
            material_properties (MaterialParameters): Material properties for the simulation.
            simulation_grid (SimulationGrid | None, optional): The grid for the simulation. Defaults to None,
                meaning a new `SimulationGrid` with default parameters is created.
            wrt_cycle (int, optional): Frequency of writing output files. Defaults to 5000.
            stop_iter (int, optional): Number of iterations to run the simulation. Defaults to 50000.
            seed (int, optional): Seed for the random number generator. Defaults to 42.
        """
        np.random.seed(seed)

        self.material = material_properties
        self.grid = SimulationGrid() if simulation_grid is None else simulation_grid
        self.wrt_cycle = wrt_cycle
        self.stop_iter = stop_iter

        self.grid.phi = self.material.composition + 0.02 * np.random.rand(self.grid.nx, self.grid.ny)
        self.output_dir = Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """Computes the discrete Laplacian using a 9-point stencil.

        Args:
            field (np.ndarray): The field for which to compute the Laplacian.

        Returns:
            np.ndarray: The computed Laplacian of the field.
        """
        lap = np.zeros_like(field)

        lap[1:-1, 1:-1] = (
            2.0 * (field[:-2, 1:-1] + field[2:, 1:-1] + field[1:-1, :-2] + field[1:-1, 2:])
            + field[:-2, :-2]
            + field[:-2, 2:]
            + field[2:, :-2]
            + field[2:, 2:]
            - 12.0 * field[1:-1, 1:-1]
        ) * self.grid.laplace_factor

        lap[0, :], lap[-1, :], lap[:, 0], lap[:, -1] = (
            lap[-2, :],
            lap[1, :],
            lap[:, -2],
            lap[:, 1],
        )
        return lap

    def free_energy(self, phi: np.ndarray) -> np.ndarray:
        """Computes the free energy derivative based on the polynomial coefficients.

        Args:
            phi (np.ndarray): The phase field variable.

        Returns:
            np.ndarray: The computed free energy derivative.
        """
        phi_ld = phi.astype(np.longdouble)
        result = self.material.free_energy_poly_deriv(phi_ld)
        return np.array(result, dtype=np.float64)

    def evolve(self) -> None:
        """Evolves the phase field using the Cahn-Hilliard equation."""
        lap_phi = self.laplacian(self.grid.phi)
        df = self.free_energy(self.grid.phi)
        self.grid.chem_pot = df - 2 * self.material.kappa * lap_phi
        self.grid.lap_chem_pot = self.laplacian(self.grid.chem_pot)
        self.grid.phi += self.material.mobility * self.grid.lap_chem_pot * self.grid.dt

    @requires("matplotlib", extra="plots")
    def save_plot(self, iteration: int) -> None:
        """Saves the current phase field as an image.

        Args:
            iteration (int): The current iteration number.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        im = ax.imshow(
            self.grid.phi,
            cmap="binary_r",
            origin="lower",
            extent=(0, self.grid.lx, 0, self.grid.ly),
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Composition φ")
        ax.set(xticks=[], yticks=[])
        fig.savefig(
            f"{self.output_dir}/phi_{iteration}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

    def run_simulation(self, plot: bool = False) -> None:
        """Runs the simulation for a specified number of iterations.

        Args:
            plot (bool): Whether to save plots of the phase field. Default is False.
        """
        for step in range(1, self.stop_iter + 1):
            self.evolve()
            if step % self.wrt_cycle == 0:
                print(f"Iteration {step}/{self.stop_iter}")
                if plot:
                    self.save_plot(step)
        print("Simulation finished!")
