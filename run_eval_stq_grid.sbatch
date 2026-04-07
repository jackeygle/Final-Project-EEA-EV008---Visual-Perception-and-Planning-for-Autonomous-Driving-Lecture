#!/bin/bash
# Two-stage STQ tuning on official postprocess:
#  1) quick grid on subset (max_frames)
#  2) full-val eval on best pair
#
# Usage:
#   sbatch run_eval_stq_grid.sbatch
#   CKPT_PATH=motion_deeplab_epoch_60.pth GRID_MAX_FRAMES=400 sbatch run_eval_stq_grid.sbatch
#
# Optional envs:
#   NMS_LIST="11 13 15"
#   SIGMA_LIST="5 7 9"
#   GRID_MAX_FRAMES=300
#
#SBATCH --job-name=mdlab-stq-grid
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/work/zhangx29/autonomous-visual-planning-project-latest/outputs/slurm-stq-grid-%j.out

set -euo pipefail

cd /scratch/work/zhangx29/autonomous-visual-planning-project-latest
mkdir -p outputs

export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
nvidia-smi

module purge 2>/dev/null || true
module load scicomp-python-env
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

CKPT_PATH="${CKPT_PATH:-motion_deeplab_epoch_60.pth}"
NMS_LIST="${NMS_LIST:-11 13 15}"
SIGMA_LIST="${SIGMA_LIST:-5 7 9}"
GRID_MAX_FRAMES="${GRID_MAX_FRAMES:-300}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
GRID_DIR="outputs/grid_${RUN_TAG}"
mkdir -p "${GRID_DIR}"

echo "Checkpoint: ${CKPT_PATH}"
echo "Grid nms: ${NMS_LIST}"
echo "Grid sigma: ${SIGMA_LIST}"
echo "Grid max_frames: ${GRID_MAX_FRAMES}"
echo "Grid dir: ${GRID_DIR}"

for nms in ${NMS_LIST}; do
  for sigma in ${SIGMA_LIST}; do
    out_json="${GRID_DIR}/stq_nms${nms}_sigma${sigma}_subset.json"
    echo ""
    echo "=== Grid eval nms=${nms} sigma=${sigma} ==="
    "${PYTHON_BIN}" run_step_eval.py \
      --data_root . \
      --split val \
      --ckpt "${CKPT_PATH}" \
      --postprocess official \
      --label_divisor 1000 \
      --nms_kernel "${nms}" \
      --track_sigma "${sigma}" \
      --max_frames "${GRID_MAX_FRAMES}" \
      --output_json "${out_json}"
  done
done

# Select best params by STQ on subset
read -r BEST_NMS BEST_SIGMA BEST_STQ <<EOF
$("${PYTHON_BIN}" - "${GRID_DIR}" <<'PY'
import glob, json, os, re, sys
grid_dir = sys.argv[1]
best = None
pat = re.compile(r"stq_nms(\d+)_sigma(\d+)_subset\.json$")
for p in sorted(glob.glob(os.path.join(grid_dir, "stq_nms*_sigma*_subset.json"))):
    m = pat.search(os.path.basename(p))
    if not m:
        continue
    nms, sigma = int(m.group(1)), int(m.group(2))
    with open(p, "r", encoding="utf-8") as f:
        stq = float(json.load(f)["STQ"])
    row = (stq, nms, sigma)
    if best is None or row > best:
        best = row
if best is None:
    raise SystemExit("No subset JSON results found")
print(best[1], best[2], best[0])
PY
)
EOF

echo ""
echo "Best subset params: nms=${BEST_NMS}, sigma=${BEST_SIGMA}, STQ=${BEST_STQ}"

BEST_FULL_JSON="${GRID_DIR}/stq_best_fullval_nms${BEST_NMS}_sigma${BEST_SIGMA}.json"
echo "=== Full-val eval for best params ==="
"${PYTHON_BIN}" run_step_eval.py \
  --data_root . \
  --split val \
  --ckpt "${CKPT_PATH}" \
  --postprocess official \
  --label_divisor 1000 \
  --nms_kernel "${BEST_NMS}" \
  --track_sigma "${BEST_SIGMA}" \
  --output_json "${BEST_FULL_JSON}"

# Copy final full result to stable path for convenience.
cp "${BEST_FULL_JSON}" outputs/stq_results_official_epoch60_gridbest.json

echo ""
echo "Grid done."
echo "Best subset params: nms=${BEST_NMS}, sigma=${BEST_SIGMA}, STQ=${BEST_STQ}"
echo "Best full result: ${BEST_FULL_JSON}"
echo "Copied to: outputs/stq_results_official_epoch60_gridbest.json"
