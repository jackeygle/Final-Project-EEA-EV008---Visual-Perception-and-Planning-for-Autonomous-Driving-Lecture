#!/bin/bash
#SBATCH --job-name=mdlab-qeval
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/work/zhangx29/autonomous-visual-planning-project-latest/outputs/slurm-qeval-%j.out

set -euo pipefail

cd /scratch/work/zhangx29/autonomous-visual-planning-project-latest
mkdir -p outputs

export PYTHONUNBUFFERED=1
module purge 2>/dev/null || true
module load scicomp-python-env
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

KITTI_CKPT="./motion_deeplab_epoch_180.pth"
echo "Using checkpoint: ${KITTI_CKPT}"

echo "====== [1/2] Quick STQ eval — official (max 300 frames) ======"
"${PYTHON_BIN}" run_step_eval.py \
  --data_root . \
  --split val \
  --ckpt "${KITTI_CKPT}" \
  --postprocess official \
  --label_divisor 1000 \
  --nms_kernel 13 \
  --track_sigma 9 \
  --max_frames 300 \
  --output_json outputs/stq_quick_official.json
echo "Official eval done"

echo "====== [2/2] Quick Visualization (30 frames) ======"
"${PYTHON_BIN}" eval.py \
  --data_root . \
  --ckpt "${KITTI_CKPT}" \
  --sequence 0002 \
  --num_frames 30 \
  --fps 5 \
  --output_dir outputs
echo "Visualization done"

echo "====== QUICK EVAL DONE ======"
