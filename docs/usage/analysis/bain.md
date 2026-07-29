# Bain Path

Traces the potential energy along the Bain transformation path connecting the BCC and FCC structures through a continuous tetragonal distortion.

```python
from ase.build import bulk
from materialsframework.analysis import BainPathAnalyzer
from materialsframework.calculators import RandomCalculator

struct = bulk("Fe", "bcc", a=2.87, cubic=True)
calc = RandomCalculator()

bain = BainPathAnalyzer(calculator=calc)
res = bain.calculate(struct)

print(res["c_a_list"])    # c/a ratios sampled along the path
print(res["energy_list"]) # potential energy at each ratio (eV)
```

See [Theory](../../theory/analysis/bain.md) for the derivation, or the [API Reference](../../api/analysis/bain.md) for the full parameter list.
