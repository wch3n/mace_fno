#!/bin/bash
# Pilot of the heterogeneous-screening benchmark (protocol Sec. 8): MACE1 seed 17,
# then frozen linear and nonlinear MACE1-FNO, seeds 17 and 29, 20k steps each.
# The test split is copied for the final evaluation but referenced by no pilot run.
set -euo pipefail
module --force purge >/dev/null 2>&1; module load EasyBuild/2024a Python >/dev/null 2>&1
ROOT=$HOME/mace_fno_runs/hetscreen_v2
SRC=/gpfs/scratch/acad/htbase/wchen/hetscreen_data_v2_20261002
BENCH=$HOME/mxene_proj/mace_fno/benchmarks
YUK=$HOME/mace_fno_runs/yukawa_L32_kappa_production/kappa_0p5/seed17
[ -e "$ROOT" ] && { echo "$ROOT exists"; exit 1; }
mkdir -p "$ROOT/data"
cp "$SRC"/hetscreen_{train,valid,test}.xyz "$SRC"/metadata.json "$SRC"/acceptance_report.md "$ROOT/data/"

M=$ROOT/mace1_seed17
mkdir -p "$M"
python3 - "$BENCH/synthetic_yukawa/mace.yaml" "$M/config.yaml" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
c.update(name="mace1_hetscreen", seed=17, work_dir=".", log_dir="./logs", model_dir="./models",
         checkpoints_dir="./checkpoints", results_dir="./results",
         train_file="../data/hetscreen_train.xyz", valid_file="../data/hetscreen_valid.xyz",
         model="MACE", num_interactions=1, max_num_epochs=200, start_swa=150)
c.pop("test_file", None)
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
PY
cat > "$M/train.sh" <<'EOS'
#!/bin/bash
#SBATCH --job-name=hs_mace1_s17
#SBATCH --partition=gpu
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20gb
#SBATCH --time=08:00:00
#SBATCH --output=train.log
module --force purge
module load EasyBuild/2024a Python CUDA
ulimit -s unlimited
mace_run_train --config ./config.yaml
test -s models/mace1_hetscreen_stagetwo.model
EOS

for arch in linear nonlinear; do
  src=$YUK/fno_frozen_m11_$([ $arch = linear ] && echo lin || echo l2)_s100k_seed17/config.yaml
  for seed in 17 29; do
    D=$ROOT/fno_${arch}_s20k_seed$seed
    mkdir -p "$D/cache"
    python3 - "$src" "$D/config.yaml" "$arch" "$seed" <<'PY'
import sys, yaml
src, dst, arch, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
c = yaml.safe_load(open(src))
assert c["architecture"] == arch, (c["architecture"], arch)
c.update(mace_model="../mace1_seed17/models/mace1_hetscreen_stagetwo.model",
         train_file="../data/hetscreen_train.xyz", validation_file="../data/hetscreen_valid.xyz",
         test_file=None, test_cache=None, steps=20000, energy_scale=0.5, force_scale=1.0,
         spectral_diagnostic_samples=0, seed=seed,
         checkpoint=f"./hetscreen_fno_{arch}_seed{seed}_float64.pt")
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
PY
    hours=$([ $arch = linear ] && echo 05 || echo 12)
    cat > "$D/train.sh" <<EOS
#!/bin/bash
#SBATCH --job-name=hs_fno_${arch}_s${seed}
#SBATCH --partition=gpu
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20gb
#SBATCH --time=${hours}:00:00
#SBATCH --output=train.log
module --force purge
module load EasyBuild/2024a Python CUDA
ulimit -s unlimited
mace-fno-train --config ./config.yaml
EOS
  done
done

cat > "$ROOT/submit_pilot.sh" <<'EOS'
#!/bin/bash
# MACE1 first; every FNO run waits for it.
set -euo pipefail
ROOT=$HOME/mace_fno_runs/hetscreen_v2
m=$(sbatch --parsable -D "$ROOT/mace1_seed17" "$ROOT/mace1_seed17/train.sh")
echo "mace1_seed17 -> $m"
for d in fno_linear_s20k_seed17 fno_linear_s20k_seed29 fno_nonlinear_s20k_seed17 fno_nonlinear_s20k_seed29; do
  j=$(sbatch --parsable --dependency=afterok:$m -D "$ROOT/$d" "$ROOT/$d/train.sh")
  echo "$d -> $j"
done
EOS
chmod +x "$ROOT/submit_pilot.sh"
echo "setup done"; ls "$ROOT" "$ROOT/data"
