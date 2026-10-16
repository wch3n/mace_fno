#!/bin/bash
# MACE1-LES on the heterogeneous-screening dataset, seeds from SEEDS (protocol Sec. 7),
# with the settings of the Yukawa MACE1-LES runs. No test file in the configs.
set -euo pipefail
module --force purge >/dev/null 2>&1; module load EasyBuild/2024a Python >/dev/null 2>&1
ROOT=$HOME/mace_fno_runs/hetscreen_v2
BENCH=$HOME/mxene_proj/mace_fno/benchmarks
for seed in ${SEEDS:-17 29 41}; do
  D=$ROOT/maceles1200_seed$seed
  [ -e "$D" ] && { echo "$D exists, skipped"; continue; }
  mkdir -p "$D"
  python3 - "$BENCH/synthetic_yukawa/mace.yaml" "$D/config.yaml" "$seed" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
c.update(name="les1_hetscreen", seed=int(sys.argv[3]), work_dir=".", log_dir="./logs",
         model_dir="./models", checkpoints_dir="./checkpoints", results_dir="./results",
         train_file="../data/hetscreen_train.xyz", valid_file="../data/hetscreen_valid.xyz",
         model="MACELES", num_interactions=1, max_num_epochs=1200, start_swa=450,
         restart_latest=True)
c.pop("test_file", None)
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
PY
  cat > "$D/train.sh" <<EOS
#!/bin/bash
#SBATCH --job-name=hs_les_s${seed}
#SBATCH --partition=ia
#SBATCH --qos=ia
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20gb
#SBATCH --time=10:00:00
#SBATCH --output=train.log
module --force purge
module load EasyBuild/2024a Python CUDA
ulimit -s unlimited
mace_run_train --config ./config.yaml
test -s models/les1_hetscreen_stagetwo.model
EOS
  j=$(sbatch --parsable -D "$D" "$D/train.sh")
  echo "maceles1200_seed$seed -> $j"
done
