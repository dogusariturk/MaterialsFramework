# NewtonNet

!!! info "Optional dependency"

    `NewtonNetCalculator` requires the `newtonnet` extra plus `torch-scatter` and `torch-cluster` built for the same PyTorch version. See [Extras Needing an Additional Install Step](../../installation.md#extras-needing-an-additional-install-step) for the verified install sequence.

    === "uv"

        ```bash
        uv add "materialsframework[newtonnet]"
        uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
        uv pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.9.1+cpu.html --reinstall
        ```

    === "pip"

        ```bash
        pip install "materialsframework[newtonnet]"
        pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
        pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.9.1+cpu.html --force-reinstall
        ```

::: materialsframework.calculators.newtonnet.NewtonNetCalculator
