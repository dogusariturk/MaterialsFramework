# TrajectoryObserver

Records the state of an atomic structure at each recorded step of a relaxation or MD run.

`TrajectoryObserver` is attached as a callback to an ASE `Optimizer` (via `BaseCalculator.relax()`) or an ASE MD integrator (via `BaseMDCalculator.run()`); ASE invokes it every `interval` steps, and it appends the current energies, forces, stress, cell, and atomic positions/species to internal lists. For MD runs it additionally records temperature and velocities. It behaves like a read-only `Sequence`: indexing or iterating over it yields the per-step property tuple. It can be exported as a `pandas.DataFrame` (`as_pandas()`) or pickled to disk (`save()`).

You should not construct or attach a `TrajectoryObserver` yourself. `relax()` and `run()` already construct and attach one internally.

For `relax()`, the returned dict's `"trajectory"` key *is* the `TrajectoryObserver` instance, so you can call its methods directly on the result:

```python
calc = SomeCalculator(fmax=0.05, steps=500)
result = calc.relax(structure)

obs = result["trajectory"]
df = obs.as_pandas()
print(df["potential_energies"])
obs.save("relaxation.traj.pkl")
```

For `run()` (molecular dynamics), the recorded properties are already unpacked into plain lists on the results dict (e.g. `result["total_energy"]`, `result["forces"]`). There's no `"trajectory"` key to pull an observer instance from.

See the [API Reference](../../api/tools/trajectory.md) if you need the full list of properties a `TrajectoryObserver` records.
