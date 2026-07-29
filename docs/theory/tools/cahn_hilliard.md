# Phase Field Model (Cahn-Hilliard)

!!! info "Optional dependency"

    Phase Field Model requires `calphad`. Plotting/visualization support additionally requires `plots`.

    === "uv"

        ```bash
        uv add "materialsframework[calphad]"

        # For plotting/visualization support
        uv add "materialsframework[calphad,plots]"
        ```

    === "pip"

        ```bash
        pip install "materialsframework[calphad]"

        # For plotting/visualization support
        pip install "materialsframework[calphad,plots]"
        ```

Simulates phase separation on a 2D grid by solving the Cahn-Hilliard equation, with the local free energy landscape imported directly from a CALPHAD database.

## Overview

Three classes work together: `SimulationGrid` holds the mesh and the state arrays (composition field, chemical potential, and their Laplacians); `MaterialParameters` fits a polynomial free-energy density from a CALPHAD Gibbs-energy calculation at a fixed composition, temperature, and phase, and carries the mobility and gradient-energy coefficient; `PhaseFieldModel` owns the finite-difference solver. `evolve()` advances one explicit timestep, `run_simulation()` iterates `stop_iter` steps and periodically writes a snapshot of the composition field.

This bridges CALPHAD thermodynamics (already used by [StabilityMap](stability_map.md) and [Sqs2tdb](sqs2tdb.md)) into a spatially resolved microstructure model: instead of an ad hoc double-well polynomial, the bulk free energy density comes from a real solution thermodynamic model for the system being simulated. The solver is 2D, uses an explicit (forward Euler) time step (so `dt` must stay small relative to the grid spacing and mobility for numerical stability), and approximates periodic boundaries rather than enforcing them exactly.

## Theory

### Governing equation

The conserved order parameter \(\phi\) (local composition) evolves by the Cahn-Hilliard equation:

$$
\frac{\partial \phi}{\partial t} = M \nabla^2 \mu, \qquad
\mu = \frac{\delta F}{\delta \phi} = f'(\phi) - 2\kappa \nabla^2 \phi
$$

where \(F[\phi] = \int \big[f(\phi) + \kappa(\nabla\phi)^2\big]\, dV\) is the total free energy (a bulk term \(f(\phi)\) plus a gradient-energy penalty that sets the interfacial energy), \(M\) is the mobility, and \(\kappa\) is the gradient-energy coefficient. `evolve()` implements this with explicit time-stepping:

$$
\phi^{t+\Delta t} = \phi^{t} + M\,\Delta t\,\nabla^2\!\big(f'(\phi^{t}) - 2\kappa\nabla^2\phi^{t}\big)
$$

### Free energy from CALPHAD

Rather than an ad hoc double-well potential, \(f(\phi)\) comes from a real thermodynamic database: `MaterialParameters` uses `pycalphad.calculate()` to sample the molar Gibbs energy \(G(x)\) versus mole fraction \(x\) of `component` in `phase` at the requested `temperature`, then least-squares fits a degree-10 polynomial

$$
f(x) = \sum_{k=0}^{10} c_k x^{k}
$$

to that curve. \(f'(\phi)\) in the evolution equation is this polynomial's analytic derivative, evaluated at each grid point's local composition. `mobility` (\(M\)) and `kappa` (\(\kappa\)) are fixed constants rather than fitted quantities.

### Spatial discretization

`SimulationGrid` discretizes the domain on an `nx x ny` mesh spanning a physical size `lx x ly` (default 2 μm x 2 μm). `PhaseFieldModel.laplacian()` uses the isotropic 9-point ("cell dynamical system") stencil

$$
\nabla^2\phi_{i,j} \approx \frac{2(\phi_{i\pm1,j} + \phi_{i,j\pm1}) + \phi_{i\pm1,j\pm1} - 12\,\phi_{i,j}}{4\,\Delta x^2}
$$

which reduces the grid-anisotropy error of a plain 5-point stencil. Boundary rows/columns approximate periodicity by reusing the adjacent interior row/column's Laplacian rather than wrapping the stencil itself across the domain.

`PhaseFieldModel.__init__` seeds \(\phi\) as the target `composition` plus a small random perturbation, so `run_simulation()` starts near-uniform and lets \(f\) drive any thermodynamically unstable composition toward phase separation.

## References

- Cahn, J.W., & Hilliard, J.E. (1958). Free energy of a nonuniform system. I. Interfacial free energy. *The Journal of Chemical Physics*, 28(2), 258-267. <https://doi.org/10.1063/1.1744102>
- Oono, Y., & Puri, S. (1988). Study of phase-separation dynamics by use of cell dynamical systems. I. Modeling. *Physical Review A*, 38(1), 434-453. <https://doi.org/10.1103/PhysRevA.38.434>
