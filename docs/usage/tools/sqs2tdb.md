# PhaseForge

!!! info "Optional dependency"

    `Sqs2tdb` requires the `calphad` extra, plus the `sqs2tdb` binary (part of [ATAT](https://axelvandewalle.github.io/www-avdw/atat/)) on `PATH`, with its SQS database configured via `~/.atat.rc`.

    === "uv"

        ```bash
        uv add "materialsframework[calphad]"
        ```

    === "pip"

        ```bash
        pip install "materialsframework[calphad]"
        ```

Fits a CALPHAD thermodynamic database (`.tdb`) solution model directly from SQS energies computed with a `MaterialsFramework` calculator, bridging MLIP energetics into standard CALPHAD-format phase-stability data.

```python
from materialsframework.calculators import GraceCalculator
from materialsframework.tools.sqs2tdb import Sqs2tdb

s2t = Sqs2tdb(calculator=GraceCalculator())
s2t.fit(species=["Al", "Ni"], lattices=["FCC_A1", "BCC_A2"])

print(s2t.dbf)   # pycalphad Database, ready for StabilityMap or direct pycalphad use
```

See [Theory](../../theory/tools/sqs2tdb.md) for the derivation, or the [API Reference](../../api/tools/sqs2tdb.md) for the full parameter list.
