# Bond Lattice Parameter

Predicts the lattice parameter of an FCC, BCC, or HCP alloy from pairwise nearest-neighbor bond lengths extracted via MLIP relaxations, rather than by linearly interpolating pure-element lattice parameters.

```python
from materialsframework.calculators import RandomCalculator
from materialsframework.tools import BondLatticeParameter

model = BondLatticeParameter("fcc", ["Co", "Cr", "Fe", "Ni"], calculator=RandomCalculator())
model.calculate()  # relaxes pure + binary reference cells, populates the bond table

composition = {"Co": 0.25, "Cr": 0.25, "Fe": 0.25, "Ni": 0.25}
print(model.predict(composition))  # bond-based lattice parameter (Å)
print(model.vegard(composition))   # Vegard's-law baseline (Å)
```

See [Theory](../../theory/tools/bond_lattice_parameter.md) for the derivation, or the [API Reference](../../api/tools/bond_lattice_parameter.md) for the full parameter list.
