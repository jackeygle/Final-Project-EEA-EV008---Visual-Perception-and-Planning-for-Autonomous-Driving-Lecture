#!/bin/bash
#SBATCH --job-name=mdlab-kitti-train
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/work/zhangx29/autonomous-visual-planning-project-latest/outputs/slurm-train-latest-%j.out

set -euo pipefail

cd /scratch/work/zhangx29/autonomous-visual-planning-project-latest
mkdir -p outputs

export PYTHONUNBUFFERED=1
module purge 2>/dev/null || true
module load scicomp-python-env
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

RESUME_ARGS=()
if [[ "${USE_RESUME:-0}" == "1" ]]; then
  RESUME_ARGS=(--resume --resume_ckpt "${RESUME_CKPT:?set RESUME_CKPT}" --start_epoch "${START_EPOCH:-1}")
fi

"${PYTHON_BIN}" train.py \
  --data_root . \
  --epochs "${EPOCHS:-60}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --accumulation_steps "${ACCUM_STEPS:-4}" \
  --lr "${LR:-1e-4}" \
  --weight_decay "${WD:-1e-4}" \
  --num_workers "${NUM_WORKERS:-6}" \
  --crop_h "${CROP_H:-385}" \
  --crop_w "${CROP_W:-1249}" \
  --aux_semantic_weight "${AUX_SEM_WEIGHT:-0.4}" \
  --save_dir "${SAVE_DIR:-.}" \
  --save_every "${SAVE_EVERY:-5}" \
  "${RESUME_ARGS[@]}"
