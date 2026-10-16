#!/bin/bash
# Production frozen MACE1-FNO runs (linear and nonlinear), 100k steps, five FNO seeds on
# the seed-17 MACE1 backbone, as in the Yukawa benchmark. Configs are the pilot configs
# with steps, seed and checkpoint changed and the test file added (evaluated at the end).
#   SEEDS="17 29 41 53 67" bash setup_frozen.sh      (prepares; submit_frozen.sh submits)
set -euo pipefail
module --force purge >/dev/null 2>&1; module load EasyBuild/2024a Python >/dev/null 2>&1
SEEDS=${SEEDS:-17 29 41 53 67}
ROOT=$HOME/mace_fno_runs/hetscreen_v2
for arch in linear nonlinear; do
  for seed in $SEEDS; do
    D=$ROOT/fno_${arch}_s100k_seed$seed
    [ -e "$D" ] && { echo "$D exists, skipped"; continue; }
    mkdir -p "$D/cache"
    python3 - "$ROOT/fno_${arch}_s20k_seed17/config.yaml" "$D/config.yaml" "$arch" "$seed" <<'PY'
import sys, yaml
src, dst, arch, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
c = yaml.safe_load(open(src))
assert c["architecture"] == arch and c["mace_training"] == "frozen"
c.update(steps=100000, seed=seed, checkpoint=f"./hetscreen_fno_{arch}_seed{seed}_float64.pt",
         test_file="../data/hetscreen_test.xyz", test_cache="./cache/test-float64.pt")
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
PY
    hours=$([ $arch = linear ] && echo "0-16" || echo "2-00")
    cat > "$D/train.sh" <<EOS
#!/bin/bash
#SBATCH --job-name=hs_frozen_${arch}_s${seed}
#SBATCH --partition=ia
#SBATCH --qos=ia
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
    echo "prepared $D"
  done
done
