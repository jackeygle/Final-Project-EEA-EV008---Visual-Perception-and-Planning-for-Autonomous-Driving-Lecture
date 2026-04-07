#!/bin/bash
# Full val-set STQ / AQ / IoU (official postprocess), same as autonomous-visual-planning-project/run_step_eval.py.
# Usage:
#   sbatch run_eval_stq_full.sbatch
#   CKPT_PATH=weights/motion_deeplab_epoch_11.pth OUT_JSON=outputs/stq_results_epoch11.json sbatch run_eval_stq_full.sbatch
#
#SBATCH --job-name=mdlab-stq-full
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/work/zhangx29/autonomous-visual-planning-project-latest/outputs/slurm-stq-full-%j.out

set -euo pipefail

cd /scratch/work/zhangx29/autonomous-visual-planning-project-latest
mkdir -p outputs

export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
nvidia-smi

module purge 2>/dev/null || true
module load scicomp-python-env
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

CKPT_PATH="${CKPT_PATH:-motion_deeplab_epoch_10.pth}"
OUT_JSON="${OUT_JSON:-outputs/stq_results_official_epoch10.json}"
NMS_KERNEL="${NMS_KERNEL:-13}"
TRACK_SIGMA="${TRACK_SIGMA:-7}"

"${PYTHON_BIN}" run_step_eval.py \
  --data_root . \
  --split val \
  --ckpt "${CKPT_PATH}" \
  --postprocess official \
  --label_divisor 1000 \
  --nms_kernel "${NMS_KERNEL}" \
  --track_sigma "${TRACK_SIGMA}" \
  --output_json "${OUT_JSON}"
