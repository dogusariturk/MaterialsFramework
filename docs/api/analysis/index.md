# Analyzers

The `analysis` package implements the *Analyzer* half of `MaterialsFramework`'s analyzer-transformation pattern. Each analyzer accepts a calculator and a structure, then returns a dict of computed material properties.

## Pattern

```python
analyzer = SomeAnalyzer(calculator=calc)
results = analyzer.calculate(structure)
```

Internally, the analyzer uses a paired *Transformation* class to generate deformed or displaced structures, evaluates them with the calculator, and fits the results to extract properties.

## Available Analyzers

| Class                           | Module                    | Description                                            |
|---------------------------------|---------------------------|--------------------------------------------------------|
| `ANNNIStackingFaultAnalyzer`    | `annni`                   | ANNNI stacking fault energies                          |
| `BainPathAnalyzer`              | `bain`                    | Bain FCC-to-BCC transformation path                    |
| `CTEAnalyzer`                   | `cte`                     | NPT-MD volumetric coefficient of thermal expansion     |
| `CubicElasticConstantsAnalyzer` | `cubic_elastic_constants` | C11, C12, C44 for cubic cells                          |
| `ElasticConstantsAnalyzer`      | `elastic_constants`       | Full elastic tensor                                    |
| `EOSAnalyzer`                   | `eos`                     | Equation of state fitting                              |
| `FormationEnergyAnalyzer`       | `formation_energy`        | Formation energy per atom                              |
| `HSolubilityAnalyzer`           | `h_solubility`            | H interstitial insertion energies and solution energy  |
| `NEBAnalyzer`                   | `neb`                     | Nudged elastic band minimum energy path                |
| `PhonopyAnalyzer`               | `phonopy`                 | Total/projected phonon DOS and thermal properties      |
| `Phono3pyAnalyzer`              | `phono3py`                | Lattice thermal conductivity                           |
| `SBEAnalyzer`                   | `sbe`                     | Surface binding energies (sputtering-resistance proxy) |
| `SurfaceAnalyzer`               | `surface`                 | Slab surface energies                                  |
| `USFEAnalyzer`                  | `usfe`                    | Generalized stacking fault curve and USFE              |
