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

::: materialsframework.tools.cahn_hilliard.PhaseFieldModel

::: materialsframework.tools.cahn_hilliard.SimulationGrid

::: materialsframework.tools.cahn_hilliard.MaterialParameters
