# Equation of State

Fits energy-volume data to an equation of state (EOS) to extract equilibrium volume, equilibrium energy, and bulk modulus.

```python
from ase.build import bulk
from materialsframework.analysis import EOSAnalyzer
from materialsframework.calculators import RandomCalculator

struct = bulk("Cu", "fcc", a=3.6, cubic=True)
calc = RandomCalculator()

eos = EOSAnalyzer(calculator=calc)
res = eos.calculate(struct)

print(res["e0"])       # equilibrium energy (eV)
print(res["v0"])       # equilibrium volume (Å³)
print(res["b0"])       # bulk modulus (eV/Å³)
print(res["b0_GPa"])   # bulk modulus (GPa)
print(res["b1"])       # pressure derivative of bulk modulus
```

See [Theory](../../theory/analysis/eos.md) for the derivation, or the [API Reference](../../api/analysis/eos.md) for the full parameter list.
