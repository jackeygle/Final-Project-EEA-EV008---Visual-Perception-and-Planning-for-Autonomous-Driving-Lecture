#!/bin/bash
#SBATCH --job-name=mdlab-evalvid
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/work/zhangx29/autonomous-visual-planning-project-latest/outputs/slurm-eval-video-%j.out

set -euo pipefail

cd /scratch/work/zhangx29/autonomous-visual-planning-project-latest
mkdir -p outputs

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

echo "Host: $(hostname)"
nvidia-smi

PYTHON_BIN="/scratch/work/zhangx29/llm-security-mini-project/.venv_phase2/bin/python"

CKPT="${CKPT:-motion_deeplab_epoch_10.pth}"
OUT_MP4="${OUT_MP4:-outputs/evaluation_video.mp4}"
OUT_PNG="${OUT_PNG:-outputs/evaluation_preview.png}"

"${PYTHON_BIN}" eval.py \
  --data_root . \
  --ckpt "${CKPT}" \
  --output_video "${OUT_MP4}" \
  --preview_png "${OUT_PNG}" \
  --target_seq "${TARGET_SEQ:-0013}" \
  --num_frames "${NUM_FRAMES:-250}" \
  --fps "${FPS:-5}"
