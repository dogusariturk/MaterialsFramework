# Unstable Stacking Fault Energy (USFE)

Computes the generalized stacking fault energy (GSFE) curve for a rigid shear along a BCC slip system, and extracts the unstable stacking fault energy (USFE) as its maximum.

```python
from ase.build import bulk
from materialsframework.analysis import USFEAnalyzer
from materialsframework.calculators import RandomCalculator

struct = bulk("Fe", "bcc", a=2.87, cubic=True)
calc = RandomCalculator()

usfe = USFEAnalyzer(calculator=calc, slip_plane="110", num_steps=11)
res = usfe.calculate(struct)

print(res["usfe_mJ_m2"])             # unstable stacking fault energy (mJ/m^2)
print(res["usfe_displacement_frac"]) # displacement fraction at the USFE peak
print(res["gsfe_curve"])             # full GSFE curve
```

See [Theory](../../theory/analysis/usfe.md) for the derivation, or the [API Reference](../../api/analysis/usfe.md) for the full parameter list.
