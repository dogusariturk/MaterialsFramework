# Base Classes

The abstract base classes that define the framework's calculator contract: `BaseCalculator` for geometry optimization/single-point evaluation, `BaseMDCalculator` for molecular dynamics. Nearly every concrete calculator (`CHGNetCalculator`, `MACECalculator`, ...) implements `BaseCalculator`; most of those also implement `BaseMDCalculator` via multiple inheritance. They are independent ABCs, not a subclass relationship.

`RandomCalculator` and `VASPCalculator` are the two exceptions
that implement only `BaseCalculator`, with no MD story:

- `RandomCalculator` since it exists to exercise relax/calculate without an MD story,

- `VASPCalculator` since this framework wraps VASP for relaxation/single-point evaluation only.

`MEGNetCalculator` is a further outlier that implements neither ABC: it only exposes `calculate()`, returning a single scalar formation energy rather than the energy/forces/stress properties other calculators return.

## Overview

`BaseCalculator` wraps an ASE `Calculator` behind two entry points: `relax()` drives a structure (and optionally its cell) toward mechanical equilibrium with a chosen `OPTIMIZERS` algorithm, and `calculate()` performs a single-point evaluation with no optimization. Both accept ASE `Atoms`, pymatgen `Structure`, or pymatgen `Molecule` interchangeably, and return whatever properties the subclass declares in `AVAILABLE_PROPERTIES` (e.g. `energy`, `forces`, `stress`).

`BaseMDCalculator` wraps the same underlying ASE `Calculator` for time-domain simulation instead of optimization: `run()` integrates the equations of motion for `steps` timesteps under one of several NVE/NVT/NPT ensembles and returns per-step trajectories of energy, forces, stress, temperature, and velocities.

## Theory

### Relaxation (`.relax()`)

Relaxation searches for atomic positions (and, if `relax_cell`, cell vectors) that locally minimize the potential energy, i.e. drive the force on every atom below a tolerance:

$$
\max_i |\mathbf{F}_i| < f_{\max}, \qquad \mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i}
$$

within `steps` iterations of the chosen `optimizer` (a member of the `OPTIMIZERS` enum: BFGS, FIRE, LBFGS, and line-search/SciPy variants). `converged` records whether `fmax` was reached before the step cap. When `relax_cell` is set, the atoms are wrapped in an ASE `FrechetCellFilter`, which exposes the cell degrees of freedom to the optimizer alongside atomic positions so that both are minimized together, driving the structure toward zero net stress at the target external pressure (zero unless overridden via `params_asecellfilter`); `hydrostatic_strain` restricts that cell relaxation to isotropic volume changes only, with no shape change. `FixSymmetry` constrains the optimizer to preserve the input structure's space-group symmetry (`symprec` sets the tolerance); `fix_atoms` freezes every atomic position so only the cell (if `relax_cell`) is optimized. A `TrajectoryObserver` (see [TrajectoryObserver](../../usage/tools/trajectory.md)) records the path taken.

`calculate()` performs the same single-point property evaluation without any of the optimization machinery above: no `FrechetCellFilter`, no constraints, no trajectory.

### Molecular Dynamics (`.run()`)

Every ensemble starts by drawing initial velocities from the Maxwell-Boltzmann distribution at `temperature` (`thermalize_momenta`), then optionally removing net linear (`stationary`) and angular (`zero_rotation`) momentum so the thermostatted dynamics don't drift or spin as a rigid body.

NVE (`nve`)
:   The microcanonical ensemble: plain Velocity-Verlet integration of Newton's equations with no thermostat or barostat, conserving total energy up to timestep discretization error:

    $$ m_i\,\ddot{\mathbf r}_i = \mathbf F_i $$

Nosé-Hoover (`nvt_nose_hoover`, `npt_nose_hoover`)
:   An extended-Lagrangian thermostat (the same `MelchionnaNPT` integrator, with a barostat term enabled for the NPT variant) that introduces a friction variable \(\xi\) with its own equation of motion, damped on timescale `ttime`:

    $$ \dot{\mathbf p}_i = \mathbf F_i - \xi\,\mathbf p_i, \qquad
       \dot\xi = \frac{1}{Q}\Big(\sum_i \frac{p_i^2}{m_i} - 3Nk_BT\Big) $$

    This samples the canonical distribution exactly in the long-time limit, unlike the velocity-rescaling thermostats below. The NPT variant additionally couples the cell to a barostat variable damped on timescale set by `pfactor`, targeting `pressure`.

Nosé-Hoover chain (`nose_hoover_chain_nvt`)
:   Chains several Nosé-Hoover thermostat variables together, each damping the fluctuations of the one before it, fixing the poor ergodicity a single Nosé-Hoover thermostat can show for small or stiff systems; `ttime` sets the chain's characteristic damping time.

Langevin (`langevin`)
:   Adds a deterministic drag and a stochastic force satisfying the fluctuation-dissipation theorem directly to the equations of motion:

    $$ m_i\dot{\mathbf v}_i = \mathbf F_i - \gamma m_i \mathbf v_i + \mathbf\eta_i(t), \qquad
       \langle \eta_i(t)\,\eta_j(t')\rangle = 2\gamma m_i k_BT\,\delta_{ij}\delta(t-t') $$

    with friction coefficient \(\gamma =\) `friction`. Samples the canonical ensemble and tends to be a numerically robust default when a purely deterministic thermostat rings or fails to equilibrate.

Andersen (`andersen`)
:   Each step, stochastically redraws the velocities of a random subset of atoms from the Maxwell-Boltzmann distribution at `temperature`, with per-atom collision probability `andersen_prob`. Simple and robust, but the velocity resets make dynamical quantities (e.g. diffusion coefficients) less trustworthy than Langevin or Nosé-Hoover trajectories.

Bussi (`bussi`)
:   Canonical sampling through velocity rescaling: rescales velocities each step by a stochastically chosen factor constructed to exactly sample the canonical kinetic-energy distribution, unlike plain Berendsen rescaling below; `taut` sets the coupling relaxation time.

Berendsen (`nvt_berendsen`, `npt_berendsen`, `inhomogeneous_npt_berendsen`)
:   Rescales velocities (and, for NPT, the cell) each step toward the target temperature/pressure with an exponential relaxation time `taut`/`taup`:

    $$ \lambda = \left[1 + \frac{\Delta t}{\tau_T}\left(\frac{T_{\text{target}}}{T(t)} - 1\right)\right]^{1/2} $$

    Numerically stable and a common choice for quickly bringing a system to a target state before switching to a rigorous thermostat for production sampling, but does not rigorously sample the canonical ensemble. `inhomogeneous_npt_berendsen` lets each cell axis flagged in `mask` relax independently, useful for anisotropic stress relief.

MTK barostats (`isotropic_mtk_npt`, `mtk_npt`, `masked_mtk_npt`)
:   The Martyna-Tobias-Klein extended-Lagrangian NPT scheme, pairing a Nosé-Hoover thermostat (damping time `ttime`) with a Parrinello-Rahman-style cell barostat (damping time `taup`) that samples the isothermal-isobaric ensemble. The isotropic variant restricts the cell to uniform volume changes; the masked variant restricts the barostat to the axes set in `mask`; the general `mtk_npt` allows full anisotropic cell fluctuations.

## References

- Nosé, S. (1984). A unified formulation of the constant temperature molecular dynamics methods. *The Journal of Chemical Physics*, 81(1), 511-519. <https://doi.org/10.1063/1.447334>
- Hoover, W.G. (1985). Canonical dynamics: Equilibrium phase-space distributions. *Physical Review A*, 31(3), 1695-1697. <https://doi.org/10.1103/PhysRevA.31.1695>
- Martyna, G.J., Tobias, D.J., & Klein, M.L. (1994). Constant pressure molecular dynamics algorithms. *The Journal of Chemical Physics*, 101(5), 4177-4189. <https://doi.org/10.1063/1.467468>
- Bussi, G., Donadio, D., & Parrinello, M. (2007). Canonical sampling through velocity rescaling. *The Journal of Chemical Physics*, 126(1), 014101. <https://doi.org/10.1063/1.2408420>
- Berendsen, H.J.C., et al. (1984). Molecular dynamics with coupling to an external bath. *The Journal of Chemical Physics*, 81(8), 3684-3690. <https://doi.org/10.1063/1.448118>
- Andersen, H.C. (1980). Molecular dynamics simulations at constant pressure and/or temperature. *The Journal of Chemical Physics*, 72(4), 2384-2393. <https://doi.org/10.1063/1.439486>

See [Single-Point Calculation](../../usage/calculate.md), [Geometry Optimization](../../usage/relaxation.md), and [Molecular Dynamics](../../usage/md.md) for hands-on usage, or the [API Reference](../../api/calculators/base.md) for the full parameter list.
