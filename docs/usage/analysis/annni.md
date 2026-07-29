# ANNNI Stacking Faults

Estimates intrinsic and extrinsic stacking fault energies of an FCC-forming composition from the relaxed energies of three bulk polytypes, using the second-order axial next-nearest-neighbor Ising (ANNNI) model.

```python
from materialsframework.analysis import ANNNIStackingFaultAnalyzer
from materialsframework.calculators import RandomCalculator

annni = ANNNIStackingFaultAnalyzer(calculator=RandomCalculator())
res = annni.calculate("Ni")

print(res["isfe"])  # intrinsic stacking fault energy (eV/A^2)
print(res["esfe"])  # extrinsic stacking fault energy (eV/A^2)
```

See [Theory](../../theory/analysis/annni.md) for the derivation, or the [API Reference](../../api/analysis/annni.md) for the full parameter list.
