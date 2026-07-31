# Nudged Elastic Band

Interpolates a series of images between two endpoint structures and relaxes them into a minimum energy path (MEP), reporting the forward and reverse reaction barriers.

```python
from ase.build import bulk
from materialsframework.analysis import NEBAnalyzer
from materialsframework.calculators import CHGNetCalculator

initial = bulk("Ni", "fcc", a=3.52, cubic=True)
final = initial.copy()
final.positions[0] += [0.5, 0.5, 0.0]  # move one atom toward a neighboring site

calc = CHGNetCalculator()
neb = NEBAnalyzer(calculator=calc, n_images=5, climb=True)
res = neb.calculate(initial, final)

print(res["energies"])         # eV per image, including endpoints
print(res["barrier"])          # forward energy barrier (eV)
print(res["reverse_barrier"])  # reverse energy barrier (eV)
print(res["reaction_energy"])  # eV, final image minus initial image
print(res["converged"])
```

See [Theory](../../theory/analysis/neb.md) for the derivation, or the [API Reference](../../api/analysis/neb.md) for the full parameter list.
