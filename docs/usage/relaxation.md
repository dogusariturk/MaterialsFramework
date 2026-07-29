# Geometry Optimization

The `relax()` method, available on every calculator except `MEGNetCalculator` (which only exposes `calculate()`), performs structure optimization using ASE optimizers combined with `FrechetCellFilter` for simultaneous cell-shape and position relaxation.

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

`relax()` itself takes no configuration keyword arguments beyond the structure: `optimizer`, `fmax`, `steps`, and `relax_cell` are all set once on the calculator's constructor and reused for every `relax()` call. Supported ASE optimizers include:

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

The returned dict includes a `trajectory` key with all intermediate structures:

```python
for frame in res["trajectory"]:
    print(frame)    # pymatgen Structure at each optimization step
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
