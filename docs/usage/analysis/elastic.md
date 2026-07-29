# Elastic Constants

Computes the full second-order elastic constant tensor of a structure from stress-strain data, along with the isotropic mechanical properties derived from it.

```python
from ase.build import bulk
from materialsframework.analysis import ElasticConstantsAnalyzer
from materialsframework.calculators import GraceCalculator

struct = bulk("Cu", "fcc", a=3.6, cubic=True)
calc = GraceCalculator()

ec = ElasticConstantsAnalyzer(calculator=calc)
res = ec.calculate(struct)

print(res["C_11"], res["C_12"], res["C_44"])  # GPa, keys depend on detected symmetry
print(res["voigt_reuss_hill_bulk_modulus"])
print(res["poisson_ratio"])
```

See [Theory](../../theory/analysis/elastic.md) for the derivation, or the [API Reference](../../api/analysis/elastic.md) for the full parameter list.
