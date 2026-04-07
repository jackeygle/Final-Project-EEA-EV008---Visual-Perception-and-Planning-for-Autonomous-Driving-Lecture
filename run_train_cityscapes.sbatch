#!/bin/bash
#SBATCH --job-name=mdlab-cityscapes-train
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/work/zhangx29/autonomous-visual-planning-project-latest/outputs/slurm-cityscapes-train-%j.out

set -euo pipefail

cd /scratch/work/zhangx29/autonomous-visual-planning-project-latest
mkdir -p outputs weights

export PYTHONUNBUFFERED=1
module purge 2>/dev/null || true
module load scicomp-python-env
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

"${PYTHON_BIN}" train_cityscapes.py \
  --data_root "${CITYSCAPES_ROOT:?set CITYSCAPES_ROOT}" \
  --split "${SPLIT:-train}" \
  --epochs "${EPOCHS:-40}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --accumulation_steps "${ACCUM_STEPS:-4}" \
  --lr "${LR:-1e-4}" \
  --weight_decay "${WD:-1e-4}" \
  --num_workers "${NUM_WORKERS:-6}" \
  --image_h "${IMAGE_H:-384}" \
  --image_w "${IMAGE_W:-1248}" \
  --aux_semantic_weight "${AUX_SEM_WEIGHT:-0.4}" \
  --save_dir "${SAVE_DIR:-weights}" \
  --save_every "${SAVE_EVERY:-5}"
