# StabilityMap

!!! info "Optional dependency"

    `StabilityMap` requires `calphad`. `CoherentStabilityMap` additionally requires `sqsgen`. Plotting/visualization support additionally requires `plots`.

    === "uv"

        ```bash
        # StabilityMap
        uv add "materialsframework[calphad]"

        # CoherentStabilityMap
        uv add "materialsframework[calphad,sqsgen]"

        # For plotting/visualization support
        uv add "materialsframework[calphad,plots]"
        ```

    === "pip"

        ```bash
        # StabilityMap
        pip install "materialsframework[calphad]"

        # CoherentStabilityMap
        pip install "materialsframework[calphad,sqsgen]"

        # For plotting/visualization support
        pip install "materialsframework[calphad,plots]"
        ```

Maps thermodynamic stability and phase-separation (spinodal) regions across a composition space, using a CALPHAD database as the source of the Gibbs energy.

```python
from materialsframework.tools import StabilityMap

sm = StabilityMap("Al-Ni-Cr-Fe.tdb", elements=["AL", "NI", "CR", "FE"], phase="FCC_A1", temperature=1200)
sm.fit()

print(sm.compositions[["AL", "NI", "CR", "FE", "negative_eigenvalues"]].head())
sm.plot()  # plotting currently supports exactly four elements
```

```python
from materialsframework.calculators import GraceCalculator
from materialsframework.tools import CoherentStabilityMap

csm = CoherentStabilityMap(
    "Ti-Zr-Nb-Ta.tdb",
    elements=["TI", "ZR", "NB", "TA"],
    phase="BCC_A2",
    crystal_structure="bcc",
    calculator=GraceCalculator(),
)
csm.fit()

print(csm.compositions[["negative_eigenvalues_chem", "negative_eigenvalues_coherent"]])
```

See [Theory](../../theory/tools/stability_map.md) for the derivation, or the [API Reference](../../api/tools/stability_map.md) for the full parameter list.
