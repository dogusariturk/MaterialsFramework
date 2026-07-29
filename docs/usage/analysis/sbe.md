# Surface Binding Energy

Screens slab terminations for the lowest-surface-energy one, then computes the per-site energy needed to remove a single surface atom to an isolated reference state: a first-principles proxy for that surface's resistance to physical sputtering.

```python
from ase.build import bulk
from materialsframework.analysis import SBEAnalyzer
from materialsframework.calculators import RandomCalculator

struct = bulk("Fe", "bcc", a=2.87, cubic=True)
calc = RandomCalculator()

sbe = SBEAnalyzer(calculator=calc, max_index=1)
res = sbe.calculate(struct)

print(res["best_miller_index"])              # Miller index of the lowest-surface-energy termination
print(res["avg_surface_binding_energy"])      # eV, mean across terminations and elements
print(res["avg_surface_binding_energy_by_element"])
```

See [Theory](../../theory/analysis/sbe.md) for the derivation, or the [API Reference](../../api/analysis/sbe.md) for the full parameter list.
