# Phono3py

!!! info "Optional dependency"

    `Phono3pyAnalyzer` requires the `phono3py` extra.

    === "uv"

        ```bash
        uv add "materialsframework[phono3py]"
        ```

    === "pip"

        ```bash
        pip install "materialsframework[phono3py]"
        ```

Computes third-order (anharmonic) force constants and uses them to solve for lattice thermal conductivity.

```python
from ase.build import bulk
from materialsframework.analysis import Phono3pyAnalyzer
from materialsframework.calculators import RandomCalculator

struct = bulk("Si", "diamond", a=5.43, cubic=True)
calc = RandomCalculator()

phono3py = Phono3pyAnalyzer(calculator=calc)
res = phono3py.calculate(struct)

print(res["kappa"])  # lattice thermal conductivity tensor, W/(m*K)
```

See [Theory](../../theory/analysis/phono3py.md) for the derivation, or the [API Reference](../../api/analysis/phono3py.md) for the full parameter list.
