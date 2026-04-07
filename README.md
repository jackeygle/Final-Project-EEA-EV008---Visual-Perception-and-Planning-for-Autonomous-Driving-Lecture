# Motion-DeepLab for KITTI-STEP

Panoptic video segmentation on KITTI-STEP using Motion-DeepLab with ResNet-50 backbone.

Built on top of the [original project prototype](https://github.com/cristichitz/autonomous-visual-planning-project).

## What We Added

### New Source Files

| File | Description |
|------|-------------|
| `train_cityscapes.py` | Cityscapes semantic-only pretraining for encoder + decoder (with res4 auxiliary loss) |
| `dataset_cityscapes.py` | Cityscapes `leftImg8bit` / `gtFine` dataset loader |
| `cityscapes_labels.py` | Cityscapes label ID → train ID mapping |
| `official_postprocess.py` | DeepLab2-aligned panoptic decoding: NMS center detection, offset-based instance grouping, and greedy motion-based tracking |
| `run_step_eval.py` | Full STQ / AQ / IoU evaluation script supporting both official and legacy postprocessing |
| `stq_metric.py` | Segmentation and Tracking Quality metric (numpy, adapted from DeepLab2) |
| `tracking.py` | IoU + motion-gated Hungarian tracker (legacy alternative) |

### Key Changes to Existing Files

#### `model.py`
- Added **res4 auxiliary semantic head** (`semantic_aux_head`) for deep supervision during training, following official Panoptic-DeepLab design.

#### `loss.py`
- Added `compute_semantic_pretrain_loss()` for Cityscapes pretraining (semantic CE + optional aux loss).
- Added `generate_motion_targets()` — computes **correct** motion offset targets: for each pixel in the current frame, the offset points to the center of the **same instance in the previous frame** (not the previous frame's own center offsets).
- `compute_loss()` now accepts separate `motion_weights` (only supervise pixels with cross-frame instance correspondence).
- Center heatmap loss is now **focused on instance regions** rather than the full image, reducing gradient dilution from background pixels.
- Loss weights (`center_loss_weight`, `offset_loss_weight`, `motion_loss_weight`) are parameterized for tuning.

#### `dataset.py`
- Added **random horizontal flip** augmentation (50% probability during training, synchronized across image + panoptic maps).
- Removed experimental official-style augmentation (random scale/crop) that did not improve results.

#### `train.py`
- Uses corrected `generate_motion_targets(current_inst, prev_inst)` instead of the previous incorrect approach.
- Passes separate `motion_weights` to `compute_loss`.
- Supports configurable loss weights via CLI (`--center_loss_weight`, `--offset_loss_weight`, `--motion_loss_weight`).
- Supports `--aux_semantic_weight` for the res4 auxiliary semantic head.

### Slurm Job Scripts

| Script | Purpose |
|--------|---------|
| `run_train_cityscapes*.sbatch` | Cityscapes pretraining (various GPU configs) |
| `run_train_v100_32g.sbatch` | KITTI training on V100-32G |
| `run_train_v100_16g.sbatch` | KITTI training on V100-16G |
| `run_train_h200.sbatch` | KITTI training on H200 |
| `run_eval_stq_full.sbatch` | Full validation STQ evaluation |
| `run_eval_stq_grid.sbatch` | Grid search over NMS kernel / tracking sigma |
| `run_eval_video.sbatch` | Generate evaluation visualization video |
| `run_full_pipeline_single_job.sbatch` | **End-to-end**: Cityscapes pretrain → KITTI train → STQ eval in a single job |
| `submit_full_pipeline.sh` | Multi-job variant using Slurm `--dependency=afterok` |

## Training Pipeline

```
1. Cityscapes semantic pretrain (train_cityscapes.py)
   └── Initializes encoder + semantic decoder on 2975 Cityscapes images
2. KITTI-STEP full training (train.py --resume)
   └── Loads Cityscapes checkpoint, trains all heads (semantic, center, offset, motion)
3. STQ evaluation (run_step_eval.py)
   └── Official postprocess: NMS → offset grouping → motion tracking → STQ/AQ/IoU
```

## Quick Start

```bash
# Single-job full pipeline (Cityscapes pretrain → KITTI train → eval)
CITYSCAPES_ROOT=/path/to/cityscapes \
CITY_EPOCHS=40 KITTI_EPOCHS=60 \
sbatch run_full_pipeline_single_job.sbatch

# Or step-by-step with dependency chaining
CITYSCAPES_ROOT=/path/to/cityscapes ./submit_full_pipeline.sh
```

## Best Results (before improvements)

| Checkpoint | NMS | Sigma | STQ | AQ | IoU |
|------------|-----|-------|-----|-----|-----|
| epoch_60 (grid-best) | 13 | 9 | 0.3251 | 0.2302 | 0.4593 |
| epoch_60 | 13 | 7 | 0.3230 | 0.2271 | 0.4593 |
| epoch_70 | 13 | 9 | 0.3124 | 0.2162 | 0.4515 |

## Requirements

Tested on Triton HPC cluster with `module load scicomp-python-env` providing:
- Python 3.12, PyTorch 2.x, torchvision, numpy, scipy, PIL, matplotlib, cv2
