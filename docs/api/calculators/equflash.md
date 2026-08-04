# EquFlash

!!! info "Optional dependency"

    `EquFlashCalculator` has no `materialsframework` extra. It's only installable from its upstream git repository (as the `GGNN` package), and needs additional manual dependencies beyond that. See [Non-Extra Calculators](../../installation.md#non-extra-calculators) for the verified, platform-specific install commands.

    === "uv"

        ```bash
        uv pip install "GGNN @ git+https://github.com/SamsungDS/GGNN.git"
        uv pip install fairchem-core==1.10.0
        uv pip install torch-geometric e3nn cuequivariance==0.6.0 cuequivariance-torch==0.6.0
        uv pip install torch_scatter==2.1.2 torch_sparse==0.6.18 \
            --find-links https://data.pyg.org/whl/torch-2.4.1+cpu.html
        uv pip install "scipy<1.17.0"
        ```

    === "pip"

        ```bash
        pip install "GGNN @ git+https://github.com/SamsungDS/GGNN.git"
        pip install fairchem-core==1.10.0
        pip install torch-geometric e3nn cuequivariance==0.6.0 cuequivariance-torch==0.6.0
        pip install torch_scatter==2.1.2 torch_sparse==0.6.18 \
            --find-links https://data.pyg.org/whl/torch-2.4.1+cpu.html
        pip install "scipy<1.17.0"
        ```

::: materialsframework.calculators.equflash.EquFlashCalculator
