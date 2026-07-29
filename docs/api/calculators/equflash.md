# EquFlash

!!! info "Optional dependency"

    `EquFlashCalculator` has no `materialsframework` extra. It's only installable from its upstream git repository (as the `GGNN` package), and needs additional manual dependencies beyond that. See [Non-Extra Calculators](../../installation.md#non-extra-calculators) for the verified, platform-specific install commands.

    === "uv"

        ```bash
        uv pip install "GGNN @ git+https://github.com/SamsungDS/GGNN.git"
        ```

    === "pip"

        ```bash
        pip install "GGNN @ git+https://github.com/SamsungDS/GGNN.git"
        ```

::: materialsframework.calculators.equflash.EquFlashCalculator
