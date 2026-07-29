# H-Solubility

Evaluates the energetics of inserting a single hydrogen atom into octahedral and tetrahedral interstitial sites of a BCC host, and reports the lowest-energy (preferred) site.

```python
from ase.build import bulk
from materialsframework.analysis import HSolubilityAnalyzer
from materialsframework.calculators import RandomCalculator

struct = bulk("Fe", "bcc", a=2.87, cubic=True)
calc = RandomCalculator()

h_sol = HSolubilityAnalyzer(calculator=calc, hydrogen_reference_energy=-3.38)
res = h_sol.calculate(struct, site_types=("octahedral", "tetrahedral"))

print(res["preferred_site_type"])
print(res["solution_energy"])  # eV
```

See [Theory](../../theory/analysis/h_solubility.md) for the derivation, or the [API Reference](../../api/analysis/h_solubility.md) for the full parameter list.
