# Cluster job templates

The SLURM files in this directory reproduce the experiments documented in the
repository. By default, each script resolves `PROJECT_ROOT` from its own
location, so the repository can be cloned anywhere. An explicit
`PROJECT_ROOT=/path/to/mace_fno` environment value overrides that behavior.

Dataset and checkpoint paths under `data/` and `artifacts/` retain useful
repository-relative defaults. Site-specific external data must be supplied at
submission time:

```bash
sbatch --export=ALL,TI_ROOT=/path/to/Ti jobs/train_ti_1331_fno.slurm

sbatch --export=ALL,MACE_MODEL=/path/to/model.model,TRAIN_FILE=/path/to/train.xyz \
  jobs/train_mos2_402.slurm
```

The `#SBATCH` account, partition, module names, and GPU resources describe the
original cluster. Adjust those directives for another site.
