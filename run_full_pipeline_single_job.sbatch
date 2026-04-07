#!/bin/bash
#SBATCH --job-name=mdlab-full-1job
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/work/zhangx29/autonomous-visual-planning-project-latest/outputs/slurm-full-1job-%j.out

set -euo pipefail

cd /scratch/work/zhangx29/autonomous-visual-planning-project-latest
mkdir -p outputs weights

export PYTHONUNBUFFERED=1
module purge 2>/dev/null || true
module load scicomp-python-env
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

echo "[1/3] Cityscapes semantic pretrain"
"${PYTHON_BIN}" train_cityscapes.py \
  --data_root "${CITYSCAPES_ROOT:?set CITYSCAPES_ROOT}" \
  --split "${CITY_SPLIT:-train}" \
  --epochs "${CITY_EPOCHS:-1}" \
  --batch_size "${CITY_BATCH_SIZE:-4}" \
  --accumulation_steps "${CITY_ACCUM_STEPS:-4}" \
  --lr "${CITY_LR:-1e-4}" \
  --weight_decay "${CITY_WD:-1e-4}" \
  --num_workers "${CITY_NUM_WORKERS:-6}" \
  --image_h "${CITY_IMAGE_H:-384}" \
  --image_w "${CITY_IMAGE_W:-1248}" \
  --aux_semantic_weight "${CITY_AUX_SEM_WEIGHT:-0.4}" \
  --save_dir "${CITY_SAVE_DIR:-weights}" \
  --save_every "${CITY_SAVE_EVERY:-1}"

CITY_CKPT="${CITY_CKPT:-${CITY_SAVE_DIR:-weights}/motion_deeplab_cityscapes_epoch_${CITY_EPOCHS:-1}.pth}"
echo "Cityscapes checkpoint: ${CITY_CKPT}"

echo "[2/3] KITTI full training"
"${PYTHON_BIN}" train.py \
  --data_root . \
  --resume \
  --resume_ckpt "${CITY_CKPT}" \
  --start_epoch 1 \
  --epochs "${KITTI_EPOCHS:-1}" \
  --batch_size "${KITTI_BATCH_SIZE:-4}" \
  --accumulation_steps "${KITTI_ACCUM_STEPS:-4}" \
  --lr "${KITTI_LR:-1e-4}" \
  --weight_decay "${KITTI_WD:-1e-4}" \
  --num_workers "${KITTI_NUM_WORKERS:-6}" \
  --crop_h "${KITTI_CROP_H:-385}" \
  --crop_w "${KITTI_CROP_W:-1249}" \
  --aux_semantic_weight "${KITTI_AUX_SEM_WEIGHT:-0.4}" \
  --center_loss_weight "${CENTER_LOSS_WEIGHT:-200.0}" \
  --offset_loss_weight "${OFFSET_LOSS_WEIGHT:-0.01}" \
  --motion_loss_weight "${MOTION_LOSS_WEIGHT:-0.01}" \
  --save_dir "${KITTI_SAVE_DIR:-.}" \
  --save_every "${KITTI_SAVE_EVERY:-1}"

KITTI_CKPT="${KITTI_CKPT:-${KITTI_SAVE_DIR:-.}/motion_deeplab_epoch_${KITTI_EPOCHS:-1}.pth}"
echo "KITTI checkpoint: ${KITTI_CKPT}"

echo "[3/3] STQ evaluation"
"${PYTHON_BIN}" run_step_eval.py \
  --data_root . \
  --split val \
  --ckpt "${KITTI_CKPT}" \
  --postprocess official \
  --label_divisor 1000 \
  --nms_kernel "${NMS_KERNEL:-13}" \
  --track_sigma "${TRACK_SIGMA:-9}" \
  --output_json "${OUT_JSON:-outputs/stq_results_full_1job.json}"

echo "Done: ${OUT_JSON:-outputs/stq_results_full_1job.json}"
