# Quick Start

This guide shows the minimal code needed to evaluate and relax a crystal structure using `MaterialsFramework`.

## Single-Point Calculation

```python
from ase.build import bulk
from materialsframework.calculators import MACECalculator

# Build an FCC copper structure
struct = bulk(name="Cu", crystalstructure="fcc", a=3.6, cubic=True)

# Instantiate the calculator (model weights are downloaded on first use)
calc = MACECalculator()

# Evaluate energy, forces, and stress with no optimization
res = calc.calculate(struct)

# Inspect results
print(res["energy"])    # eV
print(res["forces"])    # numpy array, eV/Å
print(res["stress"])    # numpy array, GPa
```

The `calculate()` method returns a dict with at minimum:

| Key               | Description                                          |
|-------------------|------------------------------------------------------|
| `final_structure` | Input structure as a pymatgen `Structure`, unchanged |
| `energy`          | Total energy (eV)                                    |
| `forces`          | Forces on each atom (eV/Å)                           |
| `stress`          | Stress tensor (GPa)                                  |

## Geometry Optimization

```python
from ase.build import bulk
from materialsframework.calculators import MACECalculator

# Build an FCC copper structure
struct = bulk(name="Cu", crystalstructure="fcc", a=3.6, cubic=True)

# Instantiate the calculator
calc = MACECalculator()

# Run geometry optimization (cell shape + atomic positions)
res = calc.relax(struct)

# Inspect results
print(res["final_structure"])   # pymatgen Structure
print(res["forces"])            # numpy array, eV/Å
print(res["stress"])            # numpy array, GPa
```

The `relax()` method returns a dict with at minimum:

| Key               | Description                                 |
|-------------------|---------------------------------------------|
| `final_structure` | Relaxed structure as a pymatgen `Structure` |
| `trajectory`      | List of intermediate structures             |
| `energy`          | Final total energy (eV)                     |
| `forces`          | Forces on each atom (eV/Å)                  |
| `stress`          | Stress tensor (GPa)                         |

## Swapping Calculators

Any supported calculator implementation can be used as a drop-in replacement:

```python
from materialsframework.calculators import CHGNetCalculator

calc = CHGNetCalculator()
res = calc.relax(struct)
```

See the [Calculators API reference](api/calculators/index.md) for all available calculators and their parameters.
