# Installation

## Install MaterialsFramework

`MaterialsFramework` is published on [PyPI](https://pypi.org/project/materialsframework/). We recommend [uv](https://docs.astral.sh/uv/) for dependency management, though a plain `pip` install also works.

=== "uv"

    ```bash
    uv add materialsframework
    ```

=== "pip"

    ```bash
    pip install materialsframework
    ```

This installs the framework core (pymatgen, ASE, numpy, scipy) plus `RandomCalculator` and `VASPCalculator`, which have no heavy ML dependencies.

## Install MLIP Extras

Each MLIP has its own extra (for example `mace`, `chgnet`, `eqv2`, `petmad`). Install either one MLIP extra or a compatibility-safe group.

=== "uv"

    ```bash
    # core + one MLIP extra
    uv add "materialsframework[chgnet]"

    # core + compatible multi-MLIP stack (example)
    uv add "materialsframework[chgnet,matgl,sevennet]"
    ```

=== "pip"

    ```bash
    # core + one MLIP extra
    pip install "materialsframework[chgnet]"

    # core + compatible multi-MLIP stack (example)
    pip install "materialsframework[chgnet,matgl,sevennet]"
    ```

Before combining MLIPs, check [MLIP Conflicts](mlip-conflicts.md). Published package metadata cannot express the project-level exclusions or manually installed optional dependencies, so downstream `uv` and `pip` installs may not reject an unsupported combination.

## Supported Platforms

`MaterialsFramework`'s core dependencies (pymatgen, ASE, numpy, scipy) install broadly, but several MLIP extras (most `torch`-based backends) only resolve cleanly on:

- Linux `x86_64`
- Linux `aarch64`
- macOS `arm64` (Apple Silicon)

If you are on a different platform, installation may still be possible with manual dependency management, but it is not a supported target.

Calculator-specific native dependencies can narrow this support. In particular, PyG does not publish the EqV2 extension wheels for Linux `aarch64`; see its install note below.

## MLIP Extras Overview

| Calculator      | Extra       | Dependency spec                                            |
|-----------------|-------------|------------------------------------------------------------|
| ALIGNN          | `alignn`    | `alignn>=2025.4.1` (extra step needed; see note below)     |
| Allegro         | `allegro`   | `nequip-allegro>=0.8.3`                                    |
| AlphaNet        | `alphanet`  | `msc-alphanet>=0.1.3` (extra step needed; see note below)  |
| CHGNet          | `chgnet`    | `chgnet>=0.4.2`                                            |
| DeePMD          | `deepmd`    | `deepmd-kit[torch]>=3.1.2`                                 |
| EqNorm          | `eqnorm`    | `eqnorm>=0.1.1` (extra step needed; see note below)        |
| EquFlash        | N/A         | `GGNN` (git-only, no pip extra; see note below)            |
| EquiformerV2    | `eqv2`      | `fairchem-core>=1.10.0,<2.0` (extra step needed; see below) |
| eSEN            | `esen`      | `fairchem-core>=2.0.0`                                     |
| GPTFF           | N/A         | `gptff` (git-only, no pip extra; see note below)           |
| GRACE           | `grace`     | `tensorpotential>=0.5.7`                                   |
| HIENet          | `hienet`    | `hienet>=1.0.1` (extra step needed; see note below)        |
| M3GNet / MEGNet | `matgl`     | `matgl>=2.0.0`                                             |
| MACE            | `mace`      | `mace-torch>=0.3.15`                                       |
| MatRIS          | `matris`    | `matris>=0.0.1`                                            |
| MatterSim       | `mattersim` | `mattersim>=1.2.1`                                         |
| NequIP          | `nequip`    | `nequip>=0.17.0`                                           |
| Nequix          | `nequix`    | `nequix>=0.4.3`                                            |
| NewtonNet       | `newtonnet` | `newtonnet>=2.0.0` (extra step needed; see note below)     |
| ORB             | `orb`       | `orb-models>=0.5.5`                                        |
| PET-MAD         | `petmad`    | `upet>=0.2.1`                                              |
| PosEGNN         | N/A         | No installable package on any public index; see note below |
| SevenNet        | `sevennet`  | `sevenn>=0.12.0`                                           |
| TACE            | `tace`      | `TACE>=0.1.0`                                              |
| UMA             | `uma`       | `fairchem-core>=2.0.0`                                     |

### Extras Needing an Additional Install Step

EqNorm, HIENet, NewtonNet, and AlphaNet need `torch-scatter` (NewtonNet also needs `torch-cluster`), installed manually from PyG's wheel index after installing the extra. EqV2 also needs PyG extensions installed manually from that wheel index.

ALIGNN needs `dgl` instead, an undeclared dependency with no PyPI wheel for Python 3.12. Install it from DGL's own wheel indexes.

Run the install commands for each calculator in the order shown:

=== "ALIGNN"

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

=== "AlphaNet"

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

=== "EqNorm"

    === "uv"

        ```bash
        uv add "materialsframework[eqnorm]"
        uv pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.9.1+cpu.html --reinstall
        ```

    === "pip"

        ```bash
        pip install "materialsframework[eqnorm]"
        pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        pip install torch-scatter -f https://data.pyg.org/whl/torch-2.9.1+cpu.html --force-reinstall
        ```

=== "EqV2"

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

    !!! danger "Linux x86_64 and macOS arm64 only"

        This combination doesn't resolve on Linux `aarch64` because PyG doesn't publish wheels for these PyTorch 2.4 extensions. On that platform you need to build them from source yourself, following PyG's [installation-from-source guide](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html#installation-from-source).

=== "HIENet"

    === "uv"

        ```bash
        uv add "materialsframework[hienet]"
        uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
        uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.9.1+cpu.html --reinstall
        ```

    === "pip"

        ```bash
        pip install "materialsframework[hienet]"
        pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
        pip install torch-scatter -f https://data.pyg.org/whl/torch-2.9.1+cpu.html --force-reinstall
        ```

=== "NewtonNet"

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

### Non-Extra Calculators

=== "EquFlash"

    `EquFlashCalculator` has no `materialsframework` extra. Its upstream `GGNN` package does not install the calculator's runtime dependencies, so add them explicitly.

    The CPU commands below produce a fully importable `EquFlashCalculator` on Linux `x86_64` and macOS `arm64`. Linux `aarch64` requires source builds because PyG does not publish the matching `torch_scatter` and `torch_sparse` wheels.

    === "uv"

        ```bash
        uv pip install "GGNN @ git+https://github.com/SamsungDS/GGNN.git"
        uv pip install fairchem-core==1.10.0
        uv pip install torch-geometric e3nn cuequivariance==0.6.0 cuequivariance-torch==0.6.0
        uv pip install torch_scatter==2.1.2 torch_sparse==0.6.18 \
            --find-links https://data.pyg.org/whl/torch-2.4.1+cpu.html
        ```

    === "pip"

        ```bash
        pip install "GGNN @ git+https://github.com/SamsungDS/GGNN.git"
        pip install fairchem-core==1.10.0
        pip install torch-geometric e3nn cuequivariance==0.6.0 cuequivariance-torch==0.6.0
        pip install torch_scatter==2.1.2 torch_sparse==0.6.18 \
            --find-links https://data.pyg.org/whl/torch-2.4.1+cpu.html
        ```

=== "GPTFF"

    `GPTFFCalculator` is only installable from its upstream git repository. That revision declares `ase>=3.26,<3.29`, while MaterialsFramework requires `ase>=3.29`. Installing GPTFF normally would therefore downgrade ASE and break MaterialsFramework's MD imports. The sequence below bypasses only GPTFF's dependency metadata, then installs its non-core runtime dependencies explicitly.

    === "uv"

        ```bash
        uv pip install --no-deps "gptff @ git+https://github.com/atomly-materials-research-lab/GPTFF.git"
        uv pip install "torch>=2.0" scikit-learn psutil tqdm
        ```

    === "pip"

        ```bash
        pip install --no-deps "gptff @ git+https://github.com/atomly-materials-research-lab/GPTFF.git"
        pip install "torch>=2.0" scikit-learn psutil tqdm
        ```

=== "PosEGNN"

    `PosEGNNCalculator` is not available on any public package index. Clone the repository, add its module directory to `PYTHONPATH`, and install the runtime dependencies that the module does not declare:

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

=== "VASP"

    `VASPCalculator` is external and requires a separately installed licensed VASP binary. It wraps ASE's own `Vasp` calculator, so configuration (the `command` argument or the `ASE_VASP_COMMAND`/`VASP_COMMAND`/`VASP_SCRIPT` env vars, plus `VASP_PP_PATH` for pseudopotentials) follows ASE's conventions rather than anything `MaterialsFramework`-specific. See ASE's [VASP calculator documentation](https://docs.ase-lib.org/ase/calculators/vasp.html) for the full configuration reference.

## Development Setup

To contribute to `MaterialsFramework` itself, rather than use it as a dependency, clone the repository and use `uv sync` directly. `MaterialsFramework` uses [uv](https://docs.astral.sh/uv/) for dependency and environment management. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone the repository:

```bash
git clone https://github.com/dogusariturk/MaterialsFramework.git
cd MaterialsFramework
```

```bash
# Core (no ML extras)
uv sync

# Core + dev tools (ruff, ty, pytest, pre-commit)
uv sync --group dev

# Core + dev tools + selected MLIP extras
uv sync --group dev --extra chgnet --extra matgl --extra sevennet
```

If you synced `alignn`, `eqnorm`, `eqv2`, `hienet`, `newtonnet`, or `alphanet`, run the corresponding `uv pip install` commands from [Extras Needing an Additional Install Step](#extras-needing-an-additional-install-step) afterward. `uv sync` alone leaves EqV2 without its required PyG extensions, leaves ALIGNN without `dgl`, and may build `torch-scatter`/`torch-cluster` from source for the other four instead of selecting the matching wheels. Re-run the commands after a later `uv sync` if it removes or replaces those manually installed packages.

## Running Tests

```bash
# Unit tests
uv run pytest -m "not integration and not slow" -v

# Integration tests
uv run pytest -m integration -v
```

Integration tests (`@pytest.mark.integration`) instantiate real ML calculators, download model weights on first run, and execute short relaxations on small structures. They are slow on first run but fast on subsequent runs once weights are cached.

## Documentation

```bash
uv sync --extra docs
uv run mkdocs serve
```

Then open <http://localhost:8000>.
