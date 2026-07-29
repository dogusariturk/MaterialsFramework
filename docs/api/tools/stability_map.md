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

::: materialsframework.tools.stability_map.StabilityMap

::: materialsframework.tools.stability_map.CoherentStabilityMap
