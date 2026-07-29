# Special Quasirandom Structures

!!! info "Optional dependency"

    `SqsGenerator` requires the `sqsgen` extra.

    === "uv"

        ```bash
        uv add "materialsframework[sqsgen]"
        ```

    === "pip"

        ```bash
        pip install "materialsframework[sqsgen]"
        ```

Generates a finite supercell whose short-range order mimics that of a truly random substitutional alloy, so disordered-alloy properties can be modeled with a small, deterministic cell instead of a much larger explicitly-disordered one.

## Overview

A randomly-decorated supercell of practical size never quite reproduces the pair (and higher-order) correlations of an infinite random alloy, since finite-size sampling noise leaves residual short-range order. A Special Quasirandom Structure (SQS) is a supercell whose atomic decoration is chosen, by optimization rather than by chance, to match the target random-alloy correlation functions as closely as the cell size allows. `SqsGenerator` wraps the `sqsgenerator` optimizer: given a target composition, crystal structure, and supercell size, it searches over atomic decorations for the one whose correlation functions best match the random-alloy target.

## Theory

For a cluster \(\alpha\) on a given coordination shell (e.g. all first-nearest-neighbor pairs), the correlation function is the average, over all clusters of that type in the supercell, of the product of a site-occupation variable across the cluster's sites:

$$
\Pi_\alpha(\sigma) = \frac{1}{N_\alpha} \sum_{\alpha} \prod_{i \in \alpha} \sigma_i
$$

For a binary A-B system with concentration \(x\) of A, the exact pair-correlation value for a truly random arrangement, on any shell, is \(\Pi^{\text{random}} = (2x-1)^2\) (multi-component systems generalize this via a discretized occupation encoding rather than a single closed-form expression). `SqsGenerator` searches for the atomic decoration that minimizes the weighted deviation from this target across the requested coordination shells:

$$
\text{objective} = \sum_{k} w_k \sum_{\alpha \in \text{shell } k} \left| \Pi_\alpha(\sigma) - \Pi_\alpha^{\text{random}} \right|
$$

where \(w_k\) is the `shell_weights` entry for shell \(k\) (default \(\{1: 1.0, 2: 0.5\}\); the first-neighbor shell is weighted twice as strongly as the second). The search runs for `iterations` trial decorations, either by random swaps (`mode="random"`) or by systematic enumeration (`mode="systematic"`); the returned `objective` is the lowest value found, with 0 meaning an exact match to the random-alloy target on every requested shell.

## References

- Zunger, A., Wei, S.-H., Ferreira, L. G., & Bernard, J. E. (1990). Special quasirandom structures. *Physical Review Letters*, 65(3), 353-356. <https://doi.org/10.1103/PhysRevLett.65.353>
- Gehringer, D., Friák, M., & Holec, D. (2023). Models of configurationally-complex alloys made simple. *Computer Physics Communications*, 286, 108664. <https://doi.org/10.1016/j.cpc.2023.108664>
