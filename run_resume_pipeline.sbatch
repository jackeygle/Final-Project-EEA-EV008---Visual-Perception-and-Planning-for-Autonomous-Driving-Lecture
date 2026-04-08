#!/bin/bash
#SBATCH --job-name=mdlab-resume
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/work/zhangx29/autonomous-visual-planning-project-latest/outputs/slurm-resume-%j.out

set -euo pipefail

cd /scratch/work/zhangx29/autonomous-visual-planning-project-latest
mkdir -p outputs

export PYTHONUNBUFFERED=1
module purge 2>/dev/null || true
module load scicomp-python-env
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

RESUME_CKPT="./motion_deeplab_epoch_160.pth"
REMAINING_EPOCHS=40
START_EPOCH=161
FINAL_EPOCH=200

echo "====== [1/4] Resume KITTI training from epoch 161 → 200 ======"
"${PYTHON_BIN}" train.py \
  --data_root . \
  --resume \
  --resume_ckpt "${RESUME_CKPT}" \
  --start_epoch ${START_EPOCH} \
  --epochs ${REMAINING_EPOCHS} \
  --batch_size 4 \
  --accumulation_steps 4 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --num_workers 6 \
  --crop_h 385 \
  --crop_w 1249 \
  --aux_semantic_weight 0.4 \
  --center_loss_weight 200.0 \
  --offset_loss_weight 0.01 \
  --motion_loss_weight 0.01 \
  --top_k_percent 0.2 \
  --small_instance_weight 3.0 \
  --lr_schedule none \
  --save_dir . \
  --save_every 20

KITTI_CKPT="./motion_deeplab_epoch_${FINAL_EPOCH}.pth"
echo "KITTI checkpoint: ${KITTI_CKPT}"

echo "====== [2/4] STQ evaluation — official (motion-offset tracking) ======"
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

echo "====== [3/4] STQ evaluation — IoU tracker ======"
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

echo "====== [4/4] Visualization ======"
"${PYTHON_BIN}" eval.py \
  --data_root . \
  --ckpt "${KITTI_CKPT}" \
  --sequence 0002 \
  --num_frames 200 \
  --fps 5 \
  --output_dir outputs
echo "Visualization done"

echo "====== ALL DONE ======"
