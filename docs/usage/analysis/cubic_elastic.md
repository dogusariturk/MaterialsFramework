# Cubic Elastic Constants

Computes \(C_{11}\), \(C_{12}\), and \(C_{44}\) for a cubic (or orthogonal) cell from three targeted energy-volume/energy-strain deformation modes, rather than a general stress-strain fit.

```python
from ase.build import bulk
from materialsframework.analysis import CubicElasticConstantsAnalyzer
from materialsframework.calculators import GraceCalculator

struct = bulk("Cu", "fcc", a=3.6, cubic=True)
calc = GraceCalculator()

cubic_ec = CubicElasticConstantsAnalyzer(calculator=calc)
res = cubic_ec.calculate(struct)

print(res["C11"], res["C12"], res["C44"])  # GPa
print(res["voigt_reuss_hill_bulk_modulus"])
print(res["poisson_ratio"])
print(res["pugh_ratio"])
```

See [Theory](../../theory/analysis/cubic_elastic.md) for the derivation, or the [API Reference](../../api/analysis/cubic_elastic.md) for the full parameter list.
