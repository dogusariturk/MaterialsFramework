# Molecular Dynamics

The `run()` method on calculators that implement `BaseMDCalculator` performs molecular dynamics using ASE's integrators. The supported ensembles cover NVE, six NVT thermostats (Nosé-Hoover, Langevin, Andersen, Bussi, Berendsen, and a Nosé-Hoover chain), and six NPT/barostat variants (Nosé-Hoover, isotropic MTK, MTK, masked MTK, Berendsen, and inhomogeneous Berendsen).

## Basic NVT Example

Every keyword has a default, including `ensemble` (`"nve"`) and `temperature` (300 K), so the example below only overrides what changes for this run.

```python
from ase.build import bulk
from materialsframework.calculators import GraceCalculator

struct = bulk("Cu", "fcc", a=3.6, cubic=True)

calc = GraceCalculator(
    ensemble="nvt_nose_hoover",
    temperature=300,   # K
    verbose=True,
)
res = calc.run(structure=struct, steps=1000)

print(res["final_structure"])
print(res["temperature"])
print(res["total_energy"])
```

## Ensembles

Pass one of the keywords below as `ensemble` when constructing the calculator.

| Ensemble keyword                | Description                                                                            |
|---------------------------------|----------------------------------------------------------------------------------------|
| `"nve"`                         | Microcanonical (constant N, V, E)                                                      |
| `"nvt_nose_hoover"`             | Canonical via Nosé-Hoover thermostat                                                   |
| `"langevin"`                    | Canonical via Langevin (friction + stochastic force) thermostat                        |
| `"andersen"`                    | Canonical via Andersen stochastic collision thermostat                                 |
| `"bussi"`                       | Canonical via Bussi stochastic velocity rescaling                                      |
| `"nvt_berendsen"`               | Canonical via Berendsen velocity scaling                                               |
| `"nose_hoover_chain_nvt"`       | Canonical via a modern Nosé-Hoover chain thermostat                                    |
| `"npt_nose_hoover"`             | Isothermal-isobaric via Nosé-Hoover                                                    |
| `"isotropic_mtk_npt"`           | Isothermal-isobaric via Martyna-Tobias-Klein (MTK), isotropic volume fluctuations only |
| `"mtk_npt"`                     | Isothermal-isobaric via MTK, full anisotropic cell fluctuations                        |
| `"masked_mtk_npt"`              | Isothermal-isobaric via MTK, cell fluctuations restricted to the axes set in `mask`    |
| `"npt_berendsen"`               | NPT with Berendsen barostat                                                            |
| `"inhomogeneous_npt_berendsen"` | Inhomogeneous NPT Berendsen                                                            |

NPT ensembles also accept `pressure` (defaults to 1 atm):

```python
# NPT example
calc = GraceCalculator(
    ensemble="npt_nose_hoover",
    temperature=1000,  # K
    pressure=0.0,      # atm
)
res = calc.run(structure=struct, steps=5000)
```

## Output Dictionary

`run()` records one entry per recorded step for every array-valued key below, at the interval set by `interval`.

| Key                | Description                               |
|--------------------|-------------------------------------------|
| `final_structure`  | Final structure as pymatgen `Structure`   |
| `total_energy`     | Array of total energies per step (eV)     |
| `potential_energy` | Array of potential energies per step (eV) |
| `kinetic_energy`   | Array of kinetic energies per step (eV)   |
| `forces`           | Array of atomic forces per step (eV/A)    |
| `stresses`         | Array of stress tensors per step          |
| `temperature`      | Array of temperatures per step (K)        |
| `velocities`       | Array of atomic velocities per step       |

## Timestep and Logging

`timestep` sets the simulated time per step, in femtoseconds. `loginterval` controls how often output is written to `logfile`, which defaults to `None` and disables logging. Set `logfile` to a path and ASE's `MDLogger` writes step, time, energy, temperature, and stress to that file every `loginterval` steps.

```python
calc = GraceCalculator(
    ensemble="nvt_nose_hoover",
    temperature=300,
    timestep=2.0,       # fs
    loginterval=10,     # write the log every 10 steps
    interval=10,        # record trajectory data every 10 steps
    logfile="md.log",
)
res = calc.run(structure=struct, steps=10000)
```

## Velocity Initialization

If the input structure has no velocities set, initial velocities are drawn from a Maxwell-Boltzmann distribution at the target temperature. If you pass an `ase.Atoms` with velocities already set via `set_velocities()`/`set_momenta()`, those are used as-is instead. This only applies to `ase.Atoms` input: pymatgen `Structure`/`Molecule` carry no velocity information, and neither does `run()`'s own `final_structure` output, so re-running on either always reinitializes velocities.
