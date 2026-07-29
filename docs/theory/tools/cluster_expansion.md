# Cluster Expansion

!!! info "Optional dependency"

    `ClusterExpansion` requires the `ce` extra.

    === "uv"

        ```bash
        uv add "materialsframework[ce]"
        ```

    === "pip"

        ```bash
        pip install "materialsframework[ce]"
        ```

Fits a cluster expansion: a linear model that predicts a configuration-dependent property (e.g. mixing energy) for any atomic decoration of a fixed lattice, trained on a set of structures evaluated with a `BaseCalculator`.

## Overview

Evaluating every possible atomic arrangement of a multi-component alloy with a calculator is intractable: the configuration space grows combinatorially with the number of sites. A cluster expansion sidesteps this by fitting a compact, linear-in-parameters model on a modest training set of explicitly-calculated configurations, then evaluating that cheap model on new configurations of the same lattice (e.g. for Monte Carlo sampling of order-disorder transitions or ground-state search).`ClusterExpansion.fit()` builds an `icet` cluster space from a primitive structure, evaluates (or reuses previously-evaluated) training structures with a calculator, and regresses the effective cluster interactions with cross-validated linear regression via `trainstation`.

## Theory

A configuration \(\sigma\) (an assignment of a chemical species to every site of the lattice defined by `primitive_structure`) has predicted property:

$$
E(\sigma) = \sum_{\alpha} m_\alpha\, J_\alpha\, \langle \Phi_\alpha(\sigma) \rangle
$$

where \(\alpha\) runs over symmetry-distinct clusters (the empty cluster, single sites, pairs, triplets, ...) up to the geometric cutoffs in `cutoffs`, \(m_\alpha\) is the multiplicity of cluster \(\alpha\) per site, \(J_\alpha\) is its effective cluster interaction (ECI, the fitted parameter), and \(\langle \Phi_\alpha(\sigma) \rangle\) is the corresponding cluster correlation function: the structure-averaged value of an orthogonal site-occupation basis function evaluated over that cluster (built by `icet.ClusterSpace` from `chemical_symbols` and the lattice symmetry, at the precision set by `symprec`/`position_tolerance`). Because \(E(\sigma)\) is linear in the \(J_\alpha\), fitting is ordinary (or regularized) linear regression: `fit()` evaluates every input structure's `fit_property` with the calculator (or reads it from a pre-populated `SQLite3Database`), builds the correlation-function design matrix via `icet.StructureContainer`, and regresses \(J_\alpha\) with `trainstation.CrossValidationEstimator` using `fit_method` (e.g. `"ardr"`, `"lasso"`, `"ridge"`, `"least-squares"`), validated by `validation_method`/`n_splits`-fold cross-validation to guard against overfitting a model with many candidate clusters and a comparatively small training set. The fitted `icet.ClusterExpansion` (stored as `self.cluster_expansion`) is then the cheap surrogate: it evaluates \(E(\sigma)\) for any configuration of the same cluster space without calling the calculator again.

## References

- Ångqvist, M., Muñoz, W. A., Rahm, J. M., Fransson, E., Durniak, C., Rozyczko, P., Rod, T. H., & Erhart, P. (2019). ICET – A Python library for constructing and sampling alloy cluster expansions. *Advanced Theory and Simulations*, 2(7), 1900015. <https://doi.org/10.1002/adts.201900015>
- Sanchez, J. M., Ducastelle, F., & Gratias, D. (1984). Generalized cluster description of multicomponent systems. *Physica A*, 128(1-2), 334-350. <https://doi.org/10.1016/0378-4371(84)90096-7>
