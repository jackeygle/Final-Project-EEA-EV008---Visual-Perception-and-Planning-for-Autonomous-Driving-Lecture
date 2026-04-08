#!/bin/bash
#SBATCH --job-name=mdlab-eval
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/work/zhangx29/autonomous-visual-planning-project-latest/outputs/slurm-eval-%j.out

set -euo pipefail

cd /scratch/work/zhangx29/autonomous-visual-planning-project-latest
mkdir -p outputs

export PYTHONUNBUFFERED=1
module purge 2>/dev/null || true
module load scicomp-python-env
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

KITTI_CKPT="./motion_deeplab_epoch_180.pth"
echo "Using checkpoint: ${KITTI_CKPT}"

echo "====== [1/3] STQ evaluation — official (motion-offset tracking) ======"
"${PYTHON_BIN}" run_step_eval.py \
  --data_root . \
  --split val \
  --ckpt "${KITTI_CKPT}" \
  --postprocess official \
  --label_divisor 1000 \
  --nms_kernel 13 \
  --track_sigma 9 \
  --output_json outputs/stq_v2_official.json
echo "Official eval done"

echo "====== [2/3] STQ evaluation — IoU tracker ======"
"${PYTHON_BIN}" run_step_eval.py \
  --data_root . \
  --split val \
  --ckpt "${KITTI_CKPT}" \
  --postprocess legacy \
  --label_divisor 1000 \
  --nms_kernel 13 \
  --track_sigma 10 \
  --output_json outputs/stq_v2_iou_tracker.json
echo "IoU tracker eval done"

echo "====== [3/3] Visualization ======"
"${PYTHON_BIN}" eval.py \
  --data_root . \
  --ckpt "${KITTI_CKPT}" \
  --sequence 0002 \
  --num_frames 200 \
  --fps 5 \
  --output_dir outputs
echo "Visualization done"

echo "====== ALL DONE ======"
