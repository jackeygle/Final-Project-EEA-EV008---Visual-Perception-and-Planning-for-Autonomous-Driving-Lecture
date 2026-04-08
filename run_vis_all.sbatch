#!/bin/bash
#SBATCH --job-name=mdlab-vis
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/work/zhangx29/autonomous-visual-planning-project-latest/outputs/slurm-vis-%j.out

set -euo pipefail

cd /scratch/work/zhangx29/autonomous-visual-planning-project-latest
mkdir -p outputs/vis_all

export PYTHONUNBUFFERED=1
module purge 2>/dev/null || true
module load scicomp-python-env
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

KITTI_CKPT="./motion_deeplab_epoch_180.pth"
VAL_SEQS="0002 0006 0007 0008 0010 0013 0014 0016 0018"

for SEQ in ${VAL_SEQS}; do
  echo "====== Visualizing sequence ${SEQ} ======"
  "${PYTHON_BIN}" eval.py \
    --data_root . \
    --ckpt "${KITTI_CKPT}" \
    --sequence "${SEQ}" \
    --num_frames 9999 \
    --fps 10 \
    --output_dir "outputs/vis_all/${SEQ}"
  echo "Sequence ${SEQ} done"
done

echo "====== ALL VISUALIZATION DONE ======"
