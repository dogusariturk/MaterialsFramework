# Geometry Optimization

The standard `BaseCalculator.relax()` method performs structure optimization using ASE optimizers. Most MLIP calculators use this implementation. `MEGNetCalculator` has no `relax()` method, `RandomCalculator` is a test stub that returns the input unchanged, and `VASPCalculator` delegates optimization to VASP.

## Basic Usage

```python
from ase.build import bulk
from materialsframework.calculators import M3GNetCalculator

struct = bulk("Fe", "bcc", a=2.87, cubic=True)
calc = M3GNetCalculator()
res = calc.relax(struct)

print(res["final_structure"])
print(res["energy"])
print(res["forces"])
print(res["stress"])
```

## Optimizer Options

Core settings such as `optimizer`, `fmax`, `steps`, and `relax_cell` are set on the calculator and reused for every `relax()` call. Additional keyword arguments passed to `relax()` are forwarded to the selected ASE optimizer's constructor. Supported optimizers include:

| Optimizer           | Description                               |
|---------------------|-------------------------------------------|
| `"FIRE"`            | Fast Inertial Relaxation Engine (default) |
| `"BFGS"`            | Broyden-Fletcher-Goldfarb-Shanno          |
| `"LBFGS"`           | Limited-memory BFGS                       |
| `"MDMin"`           | Velocity-Verlet with damping              |
| `"BFGSLineSearch"`  | BFGS with line search                     |
| `"LBFGSLineSearch"` | Limited-memory BFGS with line search      |
| `"SciPyFminBFGS"`   | BFGS optimization via SciPy               |
| `"SciPyFminCG"`     | Conjugate gradient optimization via SciPy |

Pass the optimizer name as a string when constructing the calculator:

```python
calc = M3GNetCalculator(optimizer="BFGS")
res = calc.relax(struct)
```

## Convergence Criteria

Control convergence via `fmax` (force convergence, eV/Å) and `steps` (maximum steps), set on the calculator's constructor:

```python
calc = M3GNetCalculator(fmax=0.01, steps=500)
res = calc.relax(struct)
```

## Cell Relaxation

By default, `relax()` optimizes both atomic positions and cell shape. Pass `relax_cell=False` to the calculator's constructor to fix the cell:

```python
calc = M3GNetCalculator(relax_cell=False)
res = calc.relax(struct)
```

## Trajectory

For calculators using the standard implementation, the returned dict includes a `TrajectoryObserver` under the `trajectory` key. It stores structures and calculated properties at every recorded step:

```python
trajectory = res["trajectory"]
print(trajectory.atom_positions)
print(trajectory.as_pandas())
```

## Input Formats

Both pymatgen `Structure` and ASE `Atoms` objects are accepted:

```python
from pymatgen.core import Structure

# From pymatgen Structure
pmg_struct = Structure(...)
res = calc.relax(pmg_struct)

# From ASE Atoms
from ase.build import bulk
ase_atoms = bulk("Cu", "fcc", a=3.6)
res = calc.relax(ase_atoms)
```
