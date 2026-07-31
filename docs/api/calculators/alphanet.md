# AlphaNet

!!! info "Optional dependency"

    `AlphaNetCalculator` requires the `alphanet` extra plus `torch-scatter` built for the same PyTorch version. See [Extras Needing an Additional Install Step](../../installation.md#extras-needing-an-additional-install-step) for the verified install sequence.

    === "uv"

        ```bash
        uv add "materialsframework[alphanet]"
        uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
        uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.9.1+cpu.html --reinstall
        ```

    === "pip"

        ```bash
        pip install "materialsframework[alphanet]"
        pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
        pip install torch-scatter -f https://data.pyg.org/whl/torch-2.9.1+cpu.html --force-reinstall
        ```

::: materialsframework.calculators.alphanet.AlphaNetCalculator
