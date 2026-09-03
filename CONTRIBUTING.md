# Contributing to MACE-FNO

Contributions that improve the physical formulation, numerical robustness,
documentation, or reproducible benchmarks are welcome. For a substantial
change, please open an issue first so that the intended model contract and
validation evidence can be agreed before implementation.

## Development setup

Create an isolated Python environment and install the package in editable
mode:

```bash
python3 -m pip install -e '.[dev]'
```

Code paths that load or train a MACE checkpoint additionally require:

```bash
python3 -m pip install -e '.[mace,dev]'
```

## Verification

Before submitting a change, run:

```bash
python3 -m ruff check .
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
```

New numerical behavior should include a focused regression test. Changes to
particle-mesh assignment, field operators, or force evaluation should test the
relevant conservation, symmetry, translation, or finite-difference contract.
Changes to checkpoint metadata should verify both serialization and model
reconstruction.

## Benchmarks and generated files

Keep reusable implementation code under `src/mace_fno/` and system-specific
workflows under `benchmarks/<system>/`. Benchmark launchers should read their
scientific defaults from YAML configurations.

Do not run production calculations inside the repository. Set
`MACE_FNO_WORK_ROOT` to an external scratch or work directory and keep training
data, MACE models, caches, checkpoints, scheduler logs, and generated reports
there. Only the small files needed to reproduce a benchmark should be
committed.

## Pull requests

A pull request should state:

- the physical or software problem addressed;
- any change to public APIs, checkpoint schema, or numerical defaults;
- the tests and benchmark evidence used for validation;
- remaining limitations or regimes that were not tested.

Keep refactors separate from changes to scientific behavior whenever
practical. This makes both numerical review and regression diagnosis easier.
