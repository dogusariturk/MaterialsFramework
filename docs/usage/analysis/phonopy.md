# Phonopy

!!! info "Optional dependency"

    `PhonopyAnalyzer` requires the `phonopy` extra.

    === "uv"

        ```bash
        uv add "materialsframework[phonopy]"
        ```

    === "pip"

        ```bash
        pip install "materialsframework[phonopy]"
        ```

Computes harmonic phonon frequencies, density of states, and thermal properties from finite-displacement force constants.

```python
from ase.build import bulk
from materialsframework.analysis import PhonopyAnalyzer
from materialsframework.calculators import RandomCalculator

struct = bulk("Cu", "fcc", a=3.6, cubic=True)
calc = RandomCalculator()

phonopy = PhonopyAnalyzer(calculator=calc)
res = phonopy.calculate(struct)

print(res["total_dos"])
print(res["projected_dos"])
print(res["thermal_properties"])
```

See [Theory](../../theory/analysis/phonopy.md) for the derivation, or the [API Reference](../../api/analysis/phonopy.md) for the full parameter list.
