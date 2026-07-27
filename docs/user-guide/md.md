# Molecular Dynamics

The `run()` method on any calculator performs molecular dynamics using ASE integrators. Seven ensembles are supported: NVE, NVT (Nosé–Hoover / Langevin / Andersen / Bussi), NPT (Nosé–Hoover / Berendsen), and inhomogeneous NPT Berendsen.

## Basic NVT Example

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

| Ensemble keyword | Description |
|-----------------|-------------|
| `"nve"` | Microcanonical (constant N, V, E) |
| `"nvt_nose_hoover"` | Canonical via Nosé–Hoover thermostat |
| `"langevin"` | Canonical via Langevin (friction + stochastic force) thermostat |
| `"andersen"` | Canonical via Andersen stochastic collision thermostat |
| `"bussi"` | Canonical via Bussi stochastic velocity rescaling |
| `"npt_nose_hoover"` | Isothermal-isobaric via Nosé–Hoover |
| `"npt_berendsen"` | NPT with Berendsen barostat |
| `"inhomogeneous_npt_berendsen"` | Inhomogeneous NPT Berendsen |

```python
# NPT example
calc = GraceCalculator(
    ensemble="npt_nose_hoover",
    temperature=1000,  # K
    pressure=0.0,      # GPa
)
res = calc.run(structure=struct, steps=5000)
```

## Output Dictionary

| Key | Description |
|-----|-------------|
| `final_structure` | Final structure as pymatgen `Structure` |
| `total_energy` | Array of total energies per step (eV) |
| `potential_energy` | Array of potential energies per step (eV) |
| `kinetic_energy` | Array of kinetic energies per step (eV) |
| `temperature` | Array of temperatures per step (K) |
| `trajectory` | List of `TrajectoryObserver` snapshots |

## Timestep and Logging

```python
calc = GraceCalculator(
    ensemble="nvt_nose_hoover",
    temperature=300,
    timestep=2.0,       # fs
    loginterval=10,     # record every 10 steps
)
res = calc.run(structure=struct, steps=10000)
```

## Velocity Initialization

Initial velocities are drawn from a Maxwell–Boltzmann distribution at the target temperature. You can supply pre-initialized `Atoms` with velocities set, and the calculator will respect them.
