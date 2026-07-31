# EquiformerV2

!!! info "Optional dependency"

    `EqV2Calculator` requires the `eqv2` extra and additional PyG extensions. PyG publishes the required Python 3.12 wheels for Linux `x86_64` and macOS `arm64`. See [Extras Needing an Additional Install Step](../../installation.md#extras-needing-an-additional-install-step) for the verified install sequence, or [PyG's installation-from-source guide](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html#installation-from-source) for Linux `aarch64`.

    === "uv"

        ```bash
        uv add "materialsframework[eqv2]"
        uv pip install torch_scatter==2.1.2 torch_sparse==0.6.18 --find-links https://data.pyg.org/whl/torch-2.4.1+cpu.html
        ```

    === "pip"

        ```bash
        pip install "materialsframework[eqv2]"
        pip install torch_scatter==2.1.2 torch_sparse==0.6.18 --find-links https://data.pyg.org/whl/torch-2.4.1+cpu.html
        ```

::: materialsframework.calculators.eqv2.EqV2Calculator
