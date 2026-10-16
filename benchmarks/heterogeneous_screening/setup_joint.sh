#!/bin/bash
# Joint MACE1-FNO runs on the heterogeneous-screening dataset, with the Yukawa joint
# protocol: leg 1 to 100k steps from the MACE1 checkpoint with a fresh FNO, leg 2
# resumed to 200k. Configs are the Yukawa joint config with only data/backbone paths,
# sigma_E = 0.50 and spectral samples 0 changed; the test file is evaluated at the
# end of each run, as in the Yukawa runs. Nothing is submitted:
# run submit_joint.sh in each directory (or for all) when ready.
#   SEEDS="17 29 41" ARCHS="nonlinear" bash setup_joint.sh
set -euo pipefail
module --force purge >/dev/null 2>&1; module load EasyBuild/2024a Python >/dev/null 2>&1
SEEDS=${SEEDS:-17 29 41 53 67}
ARCHS=${ARCHS:-nonlinear linear}
ROOT=$HOME/mace_fno_runs/hetscreen_v2
TEMPLATE=$HOME/mace_fno_runs/yukawa_L32_kappa_production/kappa_0p5/seed17/fno_joint_m11_l2_s100k_seed17/config.leg1.yaml
for arch in $ARCHS; do
  for seed in $SEEDS; do
    D=$ROOT/fno_joint_${arch}_s200k_seed$seed
    [ -e "$D" ] && { echo "$D exists, skipped"; continue; }
    mkdir -p "$D/cache"
    python3 - "$TEMPLATE" "$D" "$arch" "$seed" <<'PY'
import sys, yaml
src, d, arch, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
c = yaml.safe_load(open(src))
assert c["mace_training"] == "joint" and c["steps"] == 100000 and c["architecture"] == "nonlinear"
ck = f"./hetscreen_fno_joint_{arch}_seed{seed}_float64.pt"
c.update(mace_model="../mace1_seed17/models/mace1_hetscreen_stagetwo.model",
         train_file="../data/hetscreen_train.xyz", validation_file="../data/hetscreen_valid.xyz",
         test_file="../data/hetscreen_test.xyz", test_cache="./cache/test-float64.pt",
         energy_scale=0.5, force_scale=1.0,
         spectral_diagnostic_samples=0, seed=seed, checkpoint=ck)
c.pop("resume", None)
if arch == "linear":   # as in the frozen linear runs
    c.update(architecture="linear", output_initialization_scale=0.0)
yaml.safe_dump(c, open(f"{d}/config.leg1.yaml", "w"), sort_keys=False)
c.update(steps=200000, resume=ck.replace(".pt", ".last.pt"))
yaml.safe_dump(c, open(f"{d}/config.leg2.yaml", "w"), sort_keys=False)
PY
    cp "$D/config.leg1.yaml" "$D/config.yaml"
    header="#!/bin/bash
#SBATCH --job-name=hs_joint_${arch}_s${seed}
#SBATCH --partition=ia
#SBATCH --qos=ia
#SBATCH --account=htbase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20gb
#SBATCH --time=2-00:00:00"
    cat > "$D/train.sh" <<EOS
$header
#SBATCH --output=train.log
module --force purge
module load EasyBuild/2024a Python CUDA
ulimit -s unlimited
mace-fno-train --config ./config.yaml
EOS
    # Leg 2: sbatch opens --output before the script runs, so this job logs to
    # extend_%j.out, archives leg 1's train.log, then writes a fresh train.log.
    cat > "$D/extend.sh" <<EOS
$header
#SBATCH --output=extend_%j.out
set -euo pipefail
module --force purge
module load EasyBuild/2024a Python CUDA
ulimit -s unlimited
last=\$(python3 -c "import yaml; print(yaml.safe_load(open('config.leg2.yaml'))['resume'])")
[ -s "\$last" ] || { echo "no \$last; leg 1 did not save a training state"; exit 1; }
leg=1; while [ -f "train.leg\$leg.log" ]; do leg=\$((leg + 1)); done
[ -f train.log ] && mv train.log "train.leg\$leg.log"
cp config.leg2.yaml config.yaml
exec > train.log 2>&1
echo "leg \$((leg + 1)) in \$(pwd): steps 200000, resume \$last"
mace-fno-train --config ./config.yaml
EOS
    cat > "$D/submit_joint.sh" <<'EOS'
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
l1=$(sbatch --parsable train.sh)
l2=$(sbatch --parsable --dependency=afterany:$l1 extend.sh)
echo "$(basename "$PWD"): leg 1 $l1, leg 2 $l2"
EOS
    chmod +x "$D/submit_joint.sh"
    echo "prepared $D"
  done
done
