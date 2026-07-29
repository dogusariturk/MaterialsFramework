# Surface Energy

Computes the surface energy of one or more slab terminations for a given Miller index.

```python
from ase.build import bulk
from materialsframework.analysis import SurfaceAnalyzer
from materialsframework.calculators import RandomCalculator

struct = bulk("Fe", "bcc", a=2.87, cubic=True)
calc = RandomCalculator()

surface = SurfaceAnalyzer(calculator=calc, miller_index=(1, 1, 0))
res = surface.calculate(struct)

print(res["bulk_energy_per_atom"])       # eV/atom
print(res["slabs"][0]["gamma"])          # surface energy (eV/Å²)
print(res["slabs"][0]["gamma_J_m2"])     # surface energy (J/m²)
```

See [Theory](../../theory/analysis/surface.md) for the derivation, or the [API Reference](../../api/analysis/surface.md) for the full parameter list.
