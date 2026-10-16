# Contributing to MACE-FNO

Contributions can address the physical model, numerical implementation,
documentation, or benchmark workflows. For a substantial change, describe
the problem, intended behavior, and validation in an issue before implementation.

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
python3 -m ruff format --check --extend-exclude '*.md' .
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Test new numerical behavior with a focused regression check. Changes to
particle-mesh assignment, field operators, or force evaluation should test the
relevant conservation, symmetry, translation, or finite-difference identities.
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

## Documentation

Start from the physical problem or the behavior of the code. Use compact,
declarative sentences and explain the mechanism before its implications.
Separate measured results from interpretation. Give the conditions and units
of a numerical comparison, and identify incomplete fits explicitly.

Use of-phrases for possessive nouns, declarative headings, and few semicolons
or trailing `, with ...` clauses. Prefer a specific verb to repeated uses of
`retain` or `remain`. Preserve command names, configuration keys, citations,
and quoted text.

Document current workflows in the benchmark overview. Label dated submission
records and archived configurations as historical, and link to their successor
when one exists. Do not rewrite recorded settings or results as though they
came from a newer model. For slabs, use **full z-mixing** and **channel-wise
height mixing** in prose. Keep the configuration identifier `z_mixing: global`
where needed to reproduce the latter.

## Pull requests

A pull request should state:

- the physical or software problem addressed.
- any change to public APIs, checkpoint schema, or numerical defaults.
- the tests and benchmark evidence used for validation.
- remaining limitations or regimes that were not tested.

Keep refactors separate from changes to scientific behavior whenever
practical. This makes both numerical review and regression diagnosis easier.
