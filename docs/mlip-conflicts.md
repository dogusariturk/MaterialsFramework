# MLIP Conflicts

Most MLIP extras install together without any issue. A handful conflict because the upstream ML packages pin incompatible versions of shared dependencies, mainly `torch` and `numpy`. This page lists exactly which combinations conflict, so you can pick a working set before running `uv add`/`pip install`.

## Conflicts by Extra

Any two extras *not* listed against each other here can be installed together. An entry of `none` means that extra has no declared conflicts at all.

| Extra       | Conflicts with                                                                      |
|-------------|-------------------------------------------------------------------------------------|
| `allegro`   | `mace`                                                                              |
| `alphanet`  | `deepmd`, `eqv2`, `esen`, `matris`, `mattersim`, `uma`                              |
| `chgnet`    | none                                                                                |
| `deepmd`    | `alphanet`, `eqv2`                                                                  |
| `eqnorm`    | `eqv2`, `mace`                                                                      |
| `eqv2`      | `alphanet`, `deepmd`, `eqnorm`, `esen`, `mace`, `matris`, `mattersim`, `orb`, `uma` |
| `esen`      | `alphanet`, `eqv2`, `mace`, `newtonnet`                                             |
| `grace`     | none                                                                                |
| `hienet`    | none                                                                                |
| `mace`      | `allegro`, `eqnorm`, `eqv2`, `esen`, `mattersim`, `nequip`, `sevennet`, `uma`       |
| `matgl`     | none                                                                                |
| `matris`    | `alphanet`, `eqv2`, `newtonnet`                                                     |
| `mattersim` | `alphanet`, `eqv2`, `mace`, `newtonnet`                                             |
| `nequip`    | `mace`                                                                              |
| `nequix`    | none                                                                                |
| `newtonnet` | `esen`, `matris`, `mattersim`, `uma`                                                |
| `orb`       | `eqv2`                                                                              |
| `petmad`    | none                                                                                |
| `sevennet`  | `mace`                                                                              |
| `tace`      | none                                                                                |
| `uma`       | `alphanet`, `eqv2`, `mace`, `newtonnet`                                             |

## Examples

=== "uv"

    ```bash
    # OK: no declared conflict between these three
    uv add "materialsframework[chgnet,matgl,sevennet]"

    # Fails: mace conflicts with sevennet
    uv add "materialsframework[mace,sevennet]"
    ```

=== "pip"

    ```bash
    # OK: no declared conflict between these three
    pip install "materialsframework[chgnet,matgl,sevennet]"

    # Fails: mace conflicts with sevennet
    pip install "materialsframework[mace,sevennet]"
    ```

## Extras Without a Pip Package

`alignn`, `gptff`, `equflash`, and `posegnn` can't be requested with `--extra` and never appear in the conflicts table above. Each has its own install path, detailed in [Non-Extra Calculators](installation.md#non-extra-calculators):

`alignn`
:   Needs a hard-pinned `torch==2.3.0` plus a matching `dgl` build from DGL's own wheel index (not PyPI). Wheels are Linux-only; macOS needs building `dgl` from source instead.

`gptff`
:   Git-only install. Its pins (`ase>=3.22.1`, `torch>=1.6`) are loose enough to coexist with every other extra in this project.

`equflash` (installed as the `GGNN` package)
:   Git-only install. Works CPU-only on Linux x86_64/macOS arm64 once you add `fairchem-core` and matching graph-learning packages by hand, since GGNN's own `requirements.txt` omits them.

`posegnn`
:   No installable package at all; add its module directory to `PYTHONPATH` instead of installing it.
