import os

import numpy as np
from pycalphad import Database, calculate
from scipy.optimize import curve_fit

__author__ = ["Doguhan Sariturk", "Vahid Attari"]
__email__ = ["dogu.sariturk@gmail.com", "attari.v@tamu.edu"]


class SimulationGrid:
    """Handles the simulation grid and phase field variables."""
    def __init__(
            self,
            nx: int = 128,
            ny: int = 128,
            Lx: float = 2e-6,
            Ly: float = 2e-6,
            dt: float = 1e-12
    ) -> None:
        """
        Initializes the simulation grid with given parameters.

        Args
            nx (int): Number of grid points in x-direction. Default is 128.
            ny (int): Number of grid points in y-direction. Default is 128.
            Lx (float): Length of the grid in x-direction. Default is 2e-6.
            Ly (float): Length of the grid in y-direction. Default is 2e-6.
            dt (float): Time step for the simulation. Default is 1e-12.
        """

        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx = self.Lx / (self.nx - 1)
        self.dt = dt

        # Precompute constant for the 9-point Laplacian stencil
        self.laplace_factor = 1.0 / (3.0 * self.dx ** 2)

        self.phi = np.zeros((nx, ny), dtype=np.float64)
        self.lap_phi = np.zeros((nx, ny), dtype=np.float64)
        self.chem_pot = np.zeros((nx, ny), dtype=np.float64)
        self.lap_chem_pot = np.zeros((nx, ny), dtype=np.float64)


class MaterialParameters:
    """Stores material parameters such as energy and kinetic properties."""
    def __init__(
            self,
            db: Database | str,
            temperature: float,
            component: str,
            composition: float,
            elements: list[str] | None = None,
            phase: str | None = None,
    ) -> None:
        """
        Initializes material parameters with given composition and potential values.

        Args
            db (Database | str): pycalphad Database object or path to the database file.
            temperature (float): Temperature in Kelvin.
            component (str): Component name.
            composition (float): Composition value.
            elements (list[str] | None): List of elements. Default is None.
            phase (str | None): Phase name. Default is None.
        """

        def energy(x, a, b, c, d, e, f, g, h, i, j, k):
            return (a * x ** 10 +
                    b * x ** 9 +
                    c * x ** 8 +
                    d * x ** 7 +
                    e * x ** 6 +
                    f * x ** 5 +
                    g * x ** 4 +
                    h * x ** 3 +
                    i * x ** 2 +
                    j * x +
                    k)

        if phase is None and len(db.phases) > 1:
            raise ValueError("Multiple phases found in the database. Please specify a phase.")

        dbf = db if isinstance(db, Database) else Database(db)
        comps = sorted(dbf.elements) if elements is None else elements
        if "/-" in comps: comps.remove("/-")
        phase = list(dbf.phases) if phase is None else phase

        Gs = calculate(dbf, comps, phase, T=temperature, output="GM")
        xs = Gs.X.sel(component=component.upper()).values.ravel()
        ys = Gs.GM.values.ravel()
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
            simulation_grid: SimulationGrid | None = None,
            material_properties: MaterialParameters | None = None,
            wrt_cycle: int = 5000,
            stop_iter: int = 50000,
            seed: int = 42
    ) -> None:
        """
        Initializes the phase field model with simulation grid and material properties.

        Args
            simulation_grid (SimulationGrid): The grid for the simulation.
            material_properties (MaterialParameters): Material properties for the simulation.
            wrt_cycle (int): Frequency of writing output files. Default is 5000.
            stop_iter (int): Number of iterations to run the simulation. Default is 50000.
            seed (int): Seed for the random number generator. Default is 42.
        """
        np.random.seed(seed)

        self.grid = SimulationGrid() if simulation_grid is None else simulation_grid
        self.material = MaterialParameters() if material_properties is None else material_properties
        self.wrt_cycle = wrt_cycle
        self.stop_iter = stop_iter

        self.grid.phi = self.material.composition + 0.02 * np.random.rand(self.grid.nx, self.grid.ny)
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Computes the discrete Laplacian using a 9-point stencil.

        Args
            field (np.ndarray): The field for which to compute the Laplacian.

        Returns
            np.ndarray: The computed Laplacian of the field.
        """
        lap = np.zeros_like(field)

        lap[1:-1, 1:-1] = (
                                  2.0 * (field[:-2, 1:-1] + field[2:, 1:-1] + field[1:-1, :-2] + field[1:-1, 2:]) +
                                  field[:-2, :-2] + field[:-2, 2:] + field[2:, :-2] + field[2:, 2:] -
                                  12.0 * field[1:-1, 1:-1]
                          ) * self.grid.laplace_factor

        lap[0, :], lap[-1, :], lap[:, 0], lap[:, -1] = lap[-2, :], lap[1, :], lap[:, -2], lap[:, 1]
        return lap

    def free_energy(self, phi: np.ndarray) -> np.ndarray:
        """
        Computes the free energy derivative based on the polynomial coefficients.

        Args
            phi (np.ndarray): The phase field variable.

        Returns
            np.ndarray: The computed free energy derivative.
        """
        phi_ld = phi.astype(np.longdouble)
        result = self.material.free_energy_poly_deriv(phi_ld)
        return np.array(result, dtype=np.float64)

    def evolve(self) -> None:
        """
        Evolves the phase field using the Cahn-Hilliard equation.
        """
        lap_phi = self.laplacian(self.grid.phi)
        df = self.free_energy(self.grid.phi)
        self.grid.chem_pot = df - 2 * self.material.kappa * lap_phi
        self.grid.lap_chem_pot = self.laplacian(self.grid.chem_pot)
        self.grid.phi += self.material.mobility * self.grid.lap_chem_pot * self.grid.dt

    def save_plot(self, iteration: int) -> None:
        """
        Saves the current phase field as an image.

        Args
            iteration (int): The current iteration number.
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow(self.grid.phi, cmap="binary_r", origin="lower",
                       extent=[0, self.grid.Lx, 0, self.grid.Ly],
                       interpolation="nearest", vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Composition φ")
        ax.set(xticks=[], yticks=[])
        fig.savefig(f"{self.output_dir}/phi_{iteration}.png", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def run_simulation(self, plot: bool = False) -> None:
        """
        Runs the simulation for a specified number of iterations.

        Args
            plot (bool): Whether to save plots of the phase field. Default is False.
        """
        for step in range(1, self.stop_iter + 1):
            self.evolve()
            if step % self.wrt_cycle == 0:
                print(f"Iteration {step}/{self.stop_iter}")
                if plot: self.save_plot(step)
        print("Simulation finished!")
