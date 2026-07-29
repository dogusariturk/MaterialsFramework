# Phase Field Model (Cahn-Hilliard)

!!! info "Optional dependency"

    Phase Field Model requires `calphad`. Plotting/visualization support additionally requires `plots`.

    === "uv"

        ```bash
        uv add "materialsframework[calphad]"

        # For plotting/visualization support
        uv add "materialsframework[calphad,plots]"
        ```

    === "pip"

        ```bash
        pip install "materialsframework[calphad]"

        # For plotting/visualization support
        pip install "materialsframework[calphad,plots]"
        ```

Simulates phase separation on a 2D grid by solving the Cahn-Hilliard equation, with the local free energy landscape imported directly from a CALPHAD database.

```python
from materialsframework.tools import MaterialParameters, PhaseFieldModel, SimulationGrid

params = MaterialParameters(
    db="Al-Ni.tdb",
    temperature=800,
    component="NI",
    composition=0.3,
    elements=["AL", "NI"],
    phase="FCC_A1",
)
grid = SimulationGrid(nx=128, ny=128)

model = PhaseFieldModel(material_properties=params, simulation_grid=grid, stop_iter=20000, wrt_cycle=2000)
model.run_simulation(plot=True)
```

See [Theory](../../theory/tools/cahn_hilliard.md) for the derivation, or the [API Reference](../../api/tools/cahn_hilliard.md) for the full parameter list.
