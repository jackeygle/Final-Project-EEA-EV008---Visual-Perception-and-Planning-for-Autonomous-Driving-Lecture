#!/usr/bin/env bash
set -euo pipefail

cd /scratch/work/zhangx29/autonomous-visual-planning-project-latest

CITY_EPOCHS="${CITY_EPOCHS:-40}"
KITTI_EPOCHS="${KITTI_EPOCHS:-60}"
CITY_CKPT="${CITY_CKPT:-weights/motion_deeplab_cityscapes_epoch_${CITY_EPOCHS}.pth}"
KITTI_CKPT="${KITTI_CKPT:-motion_deeplab_epoch_${KITTI_EPOCHS}.pth}"
OUT_JSON="${OUT_JSON:-outputs/stq_results_city${CITY_EPOCHS}_kitti${KITTI_EPOCHS}.json}"

echo "Submitting Cityscapes pretrain..."
JOB1=$(CITYSCAPES_ROOT="${CITYSCAPES_ROOT:?set CITYSCAPES_ROOT}" EPOCHS="${CITY_EPOCHS}" sbatch --parsable run_train_cityscapes_v100_32g.sbatch)
echo "Cityscapes job: ${JOB1}"

echo "Submitting KITTI train after Cityscapes..."
JOB2=$(USE_RESUME=1 RESUME_CKPT="${CITY_CKPT}" START_EPOCH=1 EPOCHS="${KITTI_EPOCHS}" sbatch --parsable --dependency=afterok:${JOB1} run_train_v100_32g.sbatch)
echo "KITTI job: ${JOB2}"

echo "Submitting eval after KITTI..."
JOB3=$(CKPT_PATH="${KITTI_CKPT}" OUT_JSON="${OUT_JSON}" NMS_KERNEL="${NMS_KERNEL:-13}" TRACK_SIGMA="${TRACK_SIGMA:-9}" sbatch --parsable --dependency=afterok:${JOB2} run_eval_stq_full.sbatch)
echo "Eval job: ${JOB3}"

echo "Logs:"
echo "  outputs/slurm-cityscapes-32g-${JOB1}.out"
echo "  outputs/slurm-train-latest-${JOB2}.out"
echo "  outputs/slurm-stq-full-${JOB3}.out"
echo "Final JSON: ${OUT_JSON}"
