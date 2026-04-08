#!/bin/bash
#SBATCH --job-name=mdlab-full-v2
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/work/zhangx29/autonomous-visual-planning-project-latest/outputs/slurm-full-v2-%j.out

set -euo pipefail

cd /scratch/work/zhangx29/autonomous-visual-planning-project-latest
mkdir -p outputs weights

export PYTHONUNBUFFERED=1
module purge 2>/dev/null || true
module load scicomp-python-env
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

echo "====== [1/4] Cityscapes FULL-BRANCH pretrain (semantic+instance) ======"
"${PYTHON_BIN}" train_cityscapes.py \
  --data_root "${CITYSCAPES_ROOT:?set CITYSCAPES_ROOT}" \
  --split "${CITY_SPLIT:-train}" \
  --panoptic \
  --epochs "${CITY_EPOCHS:-50}" \
  --batch_size "${CITY_BATCH_SIZE:-4}" \
  --accumulation_steps "${CITY_ACCUM_STEPS:-4}" \
  --lr "${CITY_LR:-1e-4}" \
  --weight_decay "${CITY_WD:-1e-4}" \
  --num_workers "${CITY_NUM_WORKERS:-6}" \
  --image_h "${CITY_IMAGE_H:-384}" \
  --image_w "${CITY_IMAGE_W:-1248}" \
  --aux_semantic_weight "${CITY_AUX_SEM_WEIGHT:-0.4}" \
  --top_k_percent "${TOP_K_PERCENT:-0.2}" \
  --lr_schedule "${CITY_LR_SCHED:-poly}" \
  --save_dir "${CITY_SAVE_DIR:-weights}" \
  --save_every "${CITY_SAVE_EVERY:-10}"

CITY_CKPT="${CITY_CKPT:-${CITY_SAVE_DIR:-weights}/motion_deeplab_cityscapes_epoch_${CITY_EPOCHS:-50}.pth}"
echo "Cityscapes checkpoint: ${CITY_CKPT}"

echo "====== [2/4] KITTI-STEP full training (with all improvements) ======"
"${PYTHON_BIN}" train.py \
  --data_root . \
  --resume \
  --resume_ckpt "${CITY_CKPT}" \
  --start_epoch 1 \
  --epochs "${KITTI_EPOCHS:-200}" \
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
  --top_k_percent "${TOP_K_PERCENT:-0.2}" \
  --small_instance_weight "${SMALL_INST_WEIGHT:-3.0}" \
  --lr_schedule "${KITTI_LR_SCHED:-poly}" \
  --save_dir "${KITTI_SAVE_DIR:-.}" \
  --save_every "${KITTI_SAVE_EVERY:-10}"

KITTI_CKPT="${KITTI_CKPT:-${KITTI_SAVE_DIR:-.}/motion_deeplab_epoch_${KITTI_EPOCHS:-200}.pth}"
echo "KITTI checkpoint: ${KITTI_CKPT}"

echo "====== [3/4] STQ evaluation — official (motion-offset tracking) ======"
"${PYTHON_BIN}" run_step_eval.py \
  --data_root . \
  --split val \
  --ckpt "${KITTI_CKPT}" \
  --postprocess official \
  --label_divisor 1000 \
  --nms_kernel "${NMS_KERNEL:-13}" \
  --track_sigma "${TRACK_SIGMA:-9}" \
  --output_json "${OUT_JSON_OFFICIAL:-outputs/stq_v2_official.json}"
echo "Official eval done: ${OUT_JSON_OFFICIAL:-outputs/stq_v2_official.json}"

echo "====== [4/5] STQ evaluation — IoU tracker (legacy, paper B1 style) ======"
"${PYTHON_BIN}" run_step_eval.py \
  --data_root . \
  --split val \
  --ckpt "${KITTI_CKPT}" \
  --postprocess legacy \
  --label_divisor 1000 \
  --nms_kernel "${NMS_KERNEL:-13}" \
  --track_sigma "${TRACK_SIGMA:-10}" \
  --output_json "${OUT_JSON_IOU:-outputs/stq_v2_iou_tracker.json}"
echo "IoU tracker eval done: ${OUT_JSON_IOU:-outputs/stq_v2_iou_tracker.json}"

echo "====== [5/5] Visualization (result PNG + video MP4) ======"
"${PYTHON_BIN}" eval.py \
  --data_root . \
  --ckpt "${KITTI_CKPT}" \
  --sequence "${VIS_SEQ:-0002}" \
  --num_frames "${VIS_FRAMES:-200}" \
  --fps "${VIS_FPS:-5}" \
  --output_dir outputs
echo "Visualization done: outputs/evaluation_result.png + outputs/evaluation_video.mp4"

echo "====== ALL DONE ======"
