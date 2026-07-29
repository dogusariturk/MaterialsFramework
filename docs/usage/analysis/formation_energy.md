# Formation Energy

Computes the formation energy per atom of a compound relative to the relaxed ground states of its constituent elements.

```python
from ase.build import bulk
from materialsframework.analysis import FormationEnergyAnalyzer
from materialsframework.calculators import RandomCalculator

struct = bulk("NiAl", crystalstructure="rocksalt", a=4.0)
calc = RandomCalculator()

fe = FormationEnergyAnalyzer(calculator=calc)
res = fe.calculate(struct)

print(res["formation_energy"])       # eV/atom
print(res["elemental_references"])   # per-element {"structure", "energy_per_atom", "is_guessed"}
```

See [Theory](../../theory/analysis/formation_energy.md) for the derivation, or the [API Reference](../../api/analysis/formation_energy.md) for the full parameter list.
