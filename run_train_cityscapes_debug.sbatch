#!/bin/bash
#SBATCH --job-name=mdlab-cityscapes-debug
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/work/zhangx29/autonomous-visual-planning-project-latest/outputs/slurm-cityscapes-debug-%j.out

set -euo pipefail
cd /scratch/work/zhangx29/autonomous-visual-planning-project-latest
mkdir -p outputs weights
module purge 2>/dev/null || true
module load scicomp-python-env
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
"${PYTHON_BIN}" train_cityscapes.py --data_root "${CITYSCAPES_ROOT:?set CITYSCAPES_ROOT}" --epochs "${EPOCHS:-1}" --batch_size "${BATCH_SIZE:-2}" --accumulation_steps "${ACCUM_STEPS:-1}" --num_workers "${NUM_WORKERS:-2}" --save_dir "${SAVE_DIR:-weights}" --save_every 1
