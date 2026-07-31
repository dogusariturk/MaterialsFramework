# PosEGNN

!!! info "Optional dependency"

    `PosEGNNCalculator` has no `materialsframework` extra or installable package on any public index. Clone the repository, add its module directory to `PYTHONPATH`, and install its undeclared runtime dependencies. See [Non-Extra Calculators](../../installation.md#non-extra-calculators) for details.

    === "uv"

        ```bash
        git clone --depth 1 https://github.com/IBM/materials.git
        export PYTHONPATH="$PWD/materials/models/pos_egnn:$PYTHONPATH"
        uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
        uv pip install torch_geometric torch_nl==0.3
        uv pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.9.1+cpu.html --reinstall
        ```

    === "pip"

        ```bash
        git clone --depth 1 https://github.com/IBM/materials.git
        export PYTHONPATH="$PWD/materials/models/pos_egnn:$PYTHONPATH"
        pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
        pip install torch_geometric torch_nl==0.3
        pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.9.1+cpu.html --force-reinstall
        ```

::: materialsframework.calculators.posegnn.PosEGNNCalculator
