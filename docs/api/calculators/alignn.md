# ALIGNN

!!! info "Optional dependency"

    `AlignnCalculator` requires the `alignn` extra plus a compatible `dgl` wheel pinned to an exact `torch` build. See [Extras Needing an Additional Install Step](../../installation.md#extras-needing-an-additional-install-step) for the verified install sequence.

    === "uv"

        ```bash
        uv add "materialsframework[alignn]"
        uv pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
        uv pip install dgl==2.2.1 torchdata==0.9.0 pyyaml \
          --find-links https://data.dgl.ai/wheels/torch-2.3/repo.html \
          --find-links https://data.dgl.ai/wheels/repo.html
        ```

    === "pip"

        ```bash
        pip install "materialsframework[alignn]"
        pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
        pip install dgl==2.2.1 torchdata==0.9.0 pyyaml \
          --find-links https://data.dgl.ai/wheels/torch-2.3/repo.html \
          --find-links https://data.dgl.ai/wheels/repo.html
        ```

::: materialsframework.calculators.alignn.AlignnCalculator
