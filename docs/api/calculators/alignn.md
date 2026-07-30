# ALIGNN

!!! info "Optional dependency"

    `AlignnCalculator` has no `materialsframework` extra. `dgl` only ships Linux wheels, pinned
    to an exact `torch` build. See [Non-Extra Calculators](../../installation.md#non-extra-calculators)
    for the verified install sequence, or [DGL's build-from-source guide](https://docs.dgl.ai/install/index.html#install-from-source)
    for macOS.

    === "uv"

        ```bash
        uv pip install torch==2.3.0
        uv pip install "dgl @ https://data.dgl.ai/wheels/torch-2.3/dgl-2.2.1-cp312-cp312-manylinux1_x86_64.whl" torchdata==0.9.0 pyyaml
        uv pip install "alignn>=2025.4.1"
        ```

    === "pip"

        ```bash
        pip install torch==2.3.0
        pip install "dgl @ https://data.dgl.ai/wheels/torch-2.3/dgl-2.2.1-cp312-cp312-manylinux1_x86_64.whl" torchdata==0.9.0 pyyaml
        pip install "alignn>=2025.4.1"
        ```

::: materialsframework.calculators.alignn.AlignnCalculator
