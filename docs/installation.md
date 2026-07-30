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

If `uv`/`pip` reports an extra conflict, choose a different combination using [MLIP Conflicts](mlip-conflicts.md).

## Supported Platforms

`MaterialsFramework`'s core dependencies (pymatgen, ASE, numpy, scipy) install broadly, but several MLIP extras (most `torch`-based backends) only resolve cleanly on:

- Linux `x86_64`
- Linux `aarch64`
- macOS `arm64` (Apple Silicon)

If you are on a different platform, installation may still be possible with manual dependency management, but it is not a supported target.

## MLIP Extras Overview

| Calculator      | Extra       | Dependency spec                                            |
|-----------------|-------------|------------------------------------------------------------|
| ALIGNN          | N/A         | `alignn` (no pip extra; see note below)                    |
| Allegro         | `allegro`   | `nequip-allegro>=0.8.3`                                    |
| AlphaNet        | `alphanet`  | `msc-alphanet>=0.1.3`                                      |
| CHGNet          | `chgnet`    | `chgnet>=0.4.2`                                            |
| DeePMD          | `deepmd`    | `deepmd-kit[torch]>=3.1.2`                                 |
| EqNorm          | `eqnorm`    | `eqnorm>=0.1.1`                                            |
| EquFlash        | N/A         | `GGNN` (git-only, no pip extra; see note below)            |
| EquiformerV2    | `eqv2`      | `fairchem-core>=1.10.0,<2.0`                               |
| eSEN            | `esen`      | `fairchem-core>=2.0.0`                                     |
| GPTFF           | N/A         | `gptff` (git-only, no pip extra; see note below)           |
| GRACE           | `grace`     | `tensorpotential>=0.5.7`                                   |
| HIENet          | `hienet`    | `hienet>=1.0.1`                                            |
| M3GNet / MEGNet | `matgl`     | `matgl>=2.0.0`                                             |
| MACE            | `mace`      | `mace-torch>=0.3.15`                                       |
| MatRIS          | `matris`    | `matris>=0.0.1`                                            |
| MatterSim       | `mattersim` | `mattersim>=1.2.1`                                         |
| NequIP          | `nequip`    | `nequip>=0.17.0`                                           |
| Nequix          | `nequix`    | `nequix>=0.4.3`                                            |
| NewtonNet       | `newtonnet` | `newtonnet>=2.0.0`                                         |
| ORB             | `orb`       | `orb-models>=0.5.5`                                        |
| PET-MAD         | `petmad`    | `upet>=0.2.1`                                              |
| PosEGNN         | N/A         | No installable package on any public index; see note below |
| SevenNet        | `sevennet`  | `sevenn>=0.12.0`                                           |
| TACE            | `tace`      | `TACE>=0.1.0`                                              |
| UMA             | `uma`       | `fairchem-core>=2.0.0`                                     |

### Non-Extra Calculators

=== "ALIGNN"

    `AlignnCalculator` has no `materialsframework` extra. `dgl` (which it needs) has no PyPI wheel
    for Python 3.12 on Linux or macOS, so it has to come from DGL's own wheel index instead, pinned
    to the exact `torch` build that wheel was compiled against:

    ```bash
    uv pip install torch==2.3.0
    uv pip install "dgl @ https://data.dgl.ai/wheels/torch-2.3/dgl-2.2.1-cp312-cp312-manylinux1_x86_64.whl" torchdata==0.9.0 pyyaml
    uv pip install "alignn>=2025.4.1"
    ```

    Verified end to end on Linux x86_64. Don't change any one of these versions without
    re-testing, since the whole chain is tightly coupled.

    !!! warning "Still failing after installing everything above?"

        If `AlignnCalculator` still raises `TypeError: 'NoneType' object is not callable`, you're
        probably missing the system `libcurl4` library (rare outside minimal containers):

        ```bash
        apt-get install -y libcurl4
        ```

    !!! info "macOS or no prebuilt wheel for your platform"

        DGL's wheel index only covers Linux. For macOS, or any platform without a prebuilt wheel,
        DGL can be built from source instead. See their
        [build-from-source guide](https://docs.dgl.ai/install/index.html#install-from-source) for
        the macOS, Linux, and Windows steps.

=== "EquFlash"

    `EquFlashCalculator` is only installable from its upstream git repository; install it manually before using `EquFlashCalculator`:

    ```bash
    uv pip install "GGNN @ git+https://github.com/SamsungDS/GGNN.git"
    ```

    !!! warning

        That command alone won't give you a working `EquFlashCalculator`. GGNN's `setup.py` pulls in no dependencies, and its code hard-imports `fairchem-core`, which isn't installed by anything above.

        The commands below use CPU builds and were confirmed to produce a fully importable `EquFlashCalculator` on Linux x86_64 and macOS arm64 (not Linux aarch64, since PyG doesn't publish `torch_scatter`/`torch_sparse` wheels for that platform):

        ```bash
        uv pip install "GGNN @ git+https://github.com/SamsungDS/GGNN.git"
        uv pip install fairchem-core==1.10.0
        uv pip install torch-geometric e3nn cuequivariance==0.6.0 cuequivariance-torch==0.6.0
        uv pip install torch_scatter==2.1.2 torch_sparse==0.6.18 \
            --find-links https://data.pyg.org/whl/torch-2.4.1+cpu.html
        ```

=== "GPTFF"

    `GPTFFCalculator` is only installable from its upstream git repository; install it manually before using
    `GPTFFCalculator`:

    ```bash
    uv pip install "gptff @ git+https://github.com/atomly-materials-research-lab/GPTFF.git"
    ```

=== "PosEGNN"

    `PosEGNNCalculator` is not available on any public package index, so it cannot be installed via pip or uv. To use it, clone the repository and add the module directory to `PYTHONPATH` manually:

    ```bash
    git clone --depth 1 https://github.com/IBM/materials.git
    export PYTHONPATH="$PWD/materials/models/pos_egnn:$PYTHONPATH"
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
