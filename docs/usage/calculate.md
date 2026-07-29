# Single-Point Calculation

The `calculate()` method on any calculator evaluates energy, forces, and other properties for a structure exactly as given, with no optimization step.

## Basic Usage

```python
from ase.build import bulk
from materialsframework.calculators import M3GNetCalculator

struct = bulk("Fe", "bcc", a=2.87, cubic=True)
calc = M3GNetCalculator()
res = calc.calculate(struct)

print(res["final_structure"])
print(res["energy"])
print(res["forces"])
print(res["stress"])
```

## calculate() vs. relax()

`calculate()` runs a single evaluation of the underlying ASE `Calculator`: no `FrechetCellFilter`, no `FixSymmetry`/`FixAtoms` constraints, no optimizer loop, no trajectory. Use it when the input structure is already at the geometry you want evaluated, such as scoring an externally generated configuration or comparing an MLIP's energy against a DFT-relaxed reference. Use `relax()` when the structure still needs to reach mechanical equilibrium.

|                            | `calculate()` | `relax()`                                     |
|----------------------------|---------------|-----------------------------------------------|
| Optimization               | None          | Runs to `fmax`/`steps` convergence            |
| Cell relaxation            | N/A           | Optional via `relax_cell`                     |
| Returned `final_structure` | Same as input | Relaxed structure                             |
| `trajectory` key           | Not present   | `TrajectoryObserver` with intermediate frames |

## Output Dictionary

| Key                                       | Description                                                      |
|-------------------------------------------|------------------------------------------------------------------|
| `final_structure`                         | Input structure as pymatgen `Structure`, unchanged               |
| Property keys from `AVAILABLE_PROPERTIES` | e.g. `energy`, `forces`, `stress`, populated from the calculator |

## Input Formats

Both pymatgen `Structure` and ASE `Atoms` objects are accepted:

```python
from pymatgen.core import Structure

# From pymatgen Structure
pmg_struct = Structure(...)
res = calc.calculate(pmg_struct)

# From ASE Atoms
from ase.build import bulk
ase_atoms = bulk("Cu", "fcc", a=3.6)
res = calc.calculate(ase_atoms)
```
