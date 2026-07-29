# Coefficient of Thermal Expansion

Estimates the volumetric coefficient of thermal expansion from NPT molecular dynamics run at several temperatures.

```python
from ase.build import bulk
from materialsframework.analysis import CTEAnalyzer
from materialsframework.calculators import GraceCalculator

struct = bulk("Cu", "fcc", a=3.6, cubic=True)
calc = GraceCalculator()  # must be MD-capable (a BaseMDCalculator subclass)

cte = CTEAnalyzer(temperatures=[300, 600, 900], calculator=calc)
res = cte.calculate(struct, steps=10000)

print(res["cte_ppm"])  # volumetric CTE, ppm/K
```

See [Theory](../../theory/analysis/cte.md) for the derivation, or the [API Reference](../../api/analysis/cte.md) for the full parameter list.
