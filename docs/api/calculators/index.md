# Calculators

`MaterialsFramework` exposes 28 calculator implementations behind a largely unified interface (26 ML-backed calculators plus `RandomCalculator` and `VASPCalculator`). Every calculator except `MEGNetCalculator` inherits from `BaseCalculator`, providing `calculate()` for single-point evaluation and `relax()` for geometry optimization; `MEGNetCalculator` implements neither ABC and only exposes `calculate()`, predicting a single scalar formation energy. Most calculators also inherit from `BaseMDCalculator`, adding `run()` for molecular dynamics; `RandomCalculator` and `VASPCalculator` are the exceptions that implement only `BaseCalculator`, with no MD story.

## Available Calculators

| Calculator   | Class                 | Backend                   |
|--------------|-----------------------|---------------------------|
| ALIGNN       | `AlignnCalculator`    | alignn                    |
| Allegro      | `AllegroCalculator`   | nequip-allegro            |
| AlphaNet     | `AlphaNetCalculator`  | msc-alphanet              |
| CHGNet       | `CHGNetCalculator`    | chgnet                    |
| DeePMD       | `DeePMDCalculator`    | deepmd-kit                |
| EqNorm       | `EqnormCalculator`    | eqnorm                    |
| EquFlash     | `EquFlashCalculator`  | N/A                       |
| EquiformerV2 | `EqV2Calculator`      | fairchem-core             |
| eSEN         | `eSENCalculator`      | fairchem-core             |
| GPTFF        | `GPTFFCalculator`     | gptff                     |
| GRACE        | `GraceCalculator`     | tensorpotential           |
| HIENet       | `HIENetCalculator`    | hienet                    |
| M3GNet       | `M3GNetCalculator`    | matgl                     |
| MACE         | `MACECalculator`      | mace-torch                |
| MatRIS       | `MatRISCalculator`    | matris                    |
| MatterSim    | `MatterSimCalculator` | mattersim                 |
| MEGNet       | `MEGNetCalculator`    | matgl                     |
| NequIP       | `NequIPCalculator`    | nequip                    |
| Nequix       | `NequixCalculator`    | nequix                    |
| NewtonNet    | `NewtonNetCalculator` | newtonnet                 |
| ORB          | `ORBCalculator`       | orb-models                |
| PET-MAD      | `PetMadCalculator`    | upet (`pet-mad-s`)        |
| PosEGNN      | `PosEGNNCalculator`   | N/A                       |
| Random       | `RandomCalculator`    | (built-in, no ML backend) |
| SevenNet     | `SevenNetCalculator`  | sevenn                    |
| TACE         | `TACECalculator`      | TACE                      |
| UMA          | `UMACalculator`       | fairchem-core             |
| VASP         | `VASPCalculator`      | VASP (external)           |

## Common Interface

All calculators except `MEGNetCalculator` expose the same `calculate()`/`relax()` methods, and most also expose `run()`. `fmax`, `steps`, `optimizer`, and `relax_cell` are set once on the calculator's constructor, not passed to `relax()` itself:

```python
calc = SomeCalculator(fmax=0.05, steps=500, optimizer="FIRE", relax_cell=True)

# Single-point evaluation
res = calc.calculate(structure)

# Geometry optimization
res = calc.relax(structure)

# Molecular dynamics (BaseMDCalculator subclasses only, e.g. not RandomCalculator or VASPCalculator)
res = calc.run(structure=structure, steps=1000)
```

See [Base Classes](base.md) for the full API of `BaseCalculator` and `BaseMDCalculator`.
