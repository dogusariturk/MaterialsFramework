# MLIP Conflicts

Most MLIP extras install together without any issue. A handful conflict because the upstream ML packages pin incompatible versions of shared dependencies, mainly `torch`, `numpy`, and PyG's compiled extensions. This page lists exactly which combinations conflict, so you can pick a working set before running `uv add`/`pip install`.

## Conflicts by MLIP

The list below folds together two kinds of conflict. Some are version ranges that `uv` itself won't combine, so it rejects them outright. Others are Torch, PyG, and DGL post-install stacks, or calculators with no pip package at all, that don't actually work together even though no resolver ever flags them. Either way, don't install both extras in the same environment. Any two MLIPs *not* listed against each other here can be installed together. An entry of `none` means that MLIP has no known conflicts at all.

| MLIP      | Extra       | Conflicts with                                                                                       |
|-----------|-------------|-------------------------------------------------------------------------------------------------------|
| ALIGNN    | `alignn`    | AlphaNet, EqNorm, EquFlash, EqV2, HIENet, NewtonNet, PosEGNN                                           |
| Allegro   | `allegro`   | MACE                                                                                                   |
| AlphaNet  | `alphanet`  | ALIGNN, DeePMD, EquFlash, EqV2, eSEN, MatRIS, MatterSim, UMA                                           |
| CHGNet    | `chgnet`    | none                                                                                                   |
| DeePMD    | `deepmd`    | AlphaNet, EquFlash, EqV2, PosEGNN                                                                      |
| EqNorm    | `eqnorm`    | ALIGNN, EquFlash, EqV2, MACE                                                                           |
| EquFlash  | N/A         | ALIGNN, AlphaNet, DeePMD, EqNorm, eSEN, HIENet, MACE, MatRIS, MatterSim, NewtonNet, ORB, PosEGNN, UMA  |
| EqV2      | `eqv2`      | ALIGNN, AlphaNet, DeePMD, EqNorm, eSEN, HIENet, MACE, MatRIS, MatterSim, NewtonNet, ORB, PosEGNN, UMA  |
| eSEN      | `esen`      | AlphaNet, EquFlash, EqV2, MACE, NewtonNet, PosEGNN                                                     |
| GPTFF     | N/A         | none                                                                                                   |
| GRACE     | `grace`     | none                                                                                                   |
| HIENet    | `hienet`    | ALIGNN, EquFlash, EqV2                                                                                 |
| M3GNet    | `matgl`     | none                                                                                                   |
| MACE      | `mace`      | Allegro, EqNorm, EquFlash, EqV2, eSEN, MatterSim, NequIP, SevenNet, UMA                                |
| MatRIS    | `matris`    | AlphaNet, EquFlash, EqV2, NewtonNet, PosEGNN                                                           |
| MatterSim | `mattersim` | AlphaNet, EquFlash, EqV2, MACE, NewtonNet, PosEGNN                                                     |
| MEGNet    | `matgl`     | none                                                                                                   |
| NequIP    | `nequip`    | MACE                                                                                                   |
| Nequix    | `nequix`    | none                                                                                                   |
| NewtonNet | `newtonnet` | ALIGNN, EquFlash, EqV2, eSEN, MatRIS, MatterSim, UMA                                                   |
| ORB       | `orb`       | EquFlash, EqV2                                                                                         |
| PetMad    | `petmad`    | none                                                                                                   |
| PosEGNN   | N/A         | ALIGNN, DeePMD, EquFlash, EqV2, eSEN, MatRIS, MatterSim, UMA                                           |
| SevenNet  | `sevennet`  | MACE                                                                                                   |
| TACE      | `tace`      | none                                                                                                   |
| UMA       | `uma`       | AlphaNet, EquFlash, EqV2, MACE, NewtonNet, PosEGNN                                                     |

Rows with `N/A` in the Extra column (EquFlash, GPTFF, PosEGNN) have no `materialsframework` extra at all. See [Calculators Without an Extra](#calculators-without-an-extra) below for how to install them.

## Examples

=== "uv"

    ```bash
    # OK: no conflict between these three
    uv add "materialsframework[chgnet,matgl,sevennet]"

    # Fails: mace conflicts with sevennet
    uv add "materialsframework[mace,sevennet]"
    ```

=== "pip"

    ```bash
    # OK: no conflict between these three
    pip install "materialsframework[chgnet,matgl,sevennet]"

    # Fails: mace conflicts with sevennet
    pip install "materialsframework[mace,sevennet]"
    ```

!!! note "Not every conflict in the table is guaranteed to be caught automatically"

    Real transitive pin clashes, like `mace-torch`'s exact `e3nn==0.4.4` pin against `sevenn`'s `e3nn>=0.5.0`, make `uv add`/`pip install` fail on their own, with no help from this project. But `tool.uv.conflicts` itself (used to keep one lockfile valid across every extra) and the post-install-stack/non-pip entries in the table above are not part of published package metadata, so a downstream `uv add` or `pip install` may resolve one of those combinations anyway even though it is unsupported. Check the table rather than relying on the installer to reject it.

## Extras Requiring Additional Setup

The following extras need packages installed after the MaterialsFramework extra. The table above cannot account for these manual changes:

`alignn`
:   Needs `torch==2.3.0` and a matching `dgl==2.2.1` wheel from DGL's wheel indexes.

`alphanet`, `eqnorm`, and `hienet`
:   Need `torch==2.9.1` and a matching `torch-scatter` wheel from PyG's wheel index.

`newtonnet`
:   Needs `torch==2.9.1` plus matching `torch-scatter` and `torch-cluster` wheels.

`eqv2`
:   Needs `torch-scatter==2.1.2` and `torch-sparse==0.6.18` wheels matching its Torch 2.4.1 dependency. Linux `aarch64` requires building those extensions from source.

Each of these installs a different, specific Torch build (or, for `eqv2`, specific compiled extensions), which is exactly why they conflict with each other and with `alignn` in the table above. See [Extras Needing an Additional Install Step](installation.md#extras-needing-an-additional-install-step) for the complete commands.

## Calculators Without an Extra

These calculators cannot be requested with a MaterialsFramework extra; they still appear in the table above for their practical/install-time conflicts, but nothing in `tool.uv.conflicts` covers them:

`equflash` (installed as `GGNN`)
:   Git-only install with manually installed runtime dependencies. It uses the same `fairchem-core==1.10.0`, Torch 2.4.1, and PyG extension stack as EqV2. It can share an environment with EqV2, M3GNet, MEGNet, and the unlisted MLIPs, but not the other conflicts shown above.

`gptff`
:   Git-only install. Its pinned revision declares `ase<3.29`, while MaterialsFramework requires `ase>=3.29`, so the documented install bypasses that dependency metadata and installs the remaining runtime dependencies explicitly. That documented dependency set resolves with every MLIP in the table; installing GPTFF normally does not.

`posegnn`
:   No installable package on a public index. Clone its repository, add the module to `PYTHONPATH`, and install its Torch 2.9.1 and PyG dependencies manually. The upstream repository instead pins ASE 3.24, Torch 2.5.1, and NumPy 1.26.4, so the compatibility results above apply specifically to the MaterialsFramework install sequence.

See [Non-Extra Calculators](installation.md#non-extra-calculators) for the verified install commands.
