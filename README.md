# Motion-DeepLab for KITTI-STEP

Panoptic video segmentation on KITTI-STEP using Motion-DeepLab with ResNet-50 backbone.

Built on top of the [original project prototype](https://github.com/cristichitz/autonomous-visual-planning-project).

## Results

### STQ Evaluation (epoch 180, full val set — 9 sequences, 2981 frames)

| Method | STQ | AQ | IoU |
|--------|-----|-----|-----|
| **Official (motion-offset tracking)** | **0.477** | **0.429** | **0.531** |

### Per-Sequence Breakdown

| Sequence | Frames | STQ | AQ | IoU |
|----------|--------|------|------|------|
| 0002 | 233 | 0.396 | 0.347 | 0.453 |
| 0006 | 270 | 0.382 | 0.352 | 0.414 |
| 0007 | 800 | 0.531 | 0.657 | 0.429 |
| 0008 | 390 | 0.457 | 0.538 | 0.389 |
| 0010 | 294 | 0.427 | 0.437 | 0.417 |
| 0013 | 340 | 0.296 | 0.226 | 0.388 |
| 0014 | 106 | 0.424 | 0.424 | 0.424 |
| 0016 | 209 | 0.339 | 0.229 | 0.503 |
| 0018 | 339 | 0.437 | 0.479 | 0.398 |

### Improvement Over Baseline

| Version | STQ | AQ | IoU |
|---------|------|------|------|
| Original baseline (epoch 60) | 0.325 | 0.230 | 0.459 |
| **Our best (epoch 180)** | **0.477** | **0.429** | **0.531** |
| Improvement | **+47%** | **+87%** | **+16%** |

## Training Pipeline

```
Stage 1: Cityscapes Pretraining (train_cityscapes.py)
   ├── Full-branch: semantic + instance center + offset
   ├── Top-K OHEM cross-entropy (hardest 20% pixels)
   └── ~50 epochs on Cityscapes train set

Stage 2: KITTI-STEP Fine-tuning (train.py)
   ├── Multi-scale augmentation (0.5x–2.0x random scale + crop + flip)
   ├── Top-K semantic loss + small instance weighting (3x for <4096px)
   ├── Correct motion offset targets (current→previous instance centers)
   ├── Deep supervision via res4 auxiliary semantic head
   ├── Mixed precision (AMP) + gradient accumulation
   └── ~180 epochs

Stage 3: STQ Evaluation (run_step_eval.py)
   ├── GPU-accelerated post-processing (NMS → offset grouping → panoptic merge)
   ├── Motion-offset tracking (official) or IoU-based Hungarian tracking (legacy)
   └── STQ / AQ / IoU metrics (DeepLab2-aligned)

Stage 4: Visualization (eval.py)
   ├── 2×2 grid: Input | Semantic Overlay | Center Heatmap | Motion Vectors
   └── PNG preview + MP4 video per sequence
```

## What We Changed vs. Original Repository

### Architecture (`model.py`)
- Added **res4 auxiliary semantic head** for deep supervision
- Backbone now outputs `{res2, res3, res4, res5}` (added res4)
- Model is now a **stateless pure encoder-decoder** — no internal post-processor or tracker
- Constructor takes no arguments (post-processing is external)

### Loss Functions (`loss.py`)
- **Top-K OHEM cross-entropy**: only backpropagates on the hardest 20% pixels
- **Small instance weighting**: 3x weight for instances < 4096 pixels
- **Thing-only center loss**: MSE computed only on thing-class pixels (not background)
- **Correct motion targets** (`generate_motion_targets`): offsets point to same-instance center in previous frame
- **Separate motion weights**: only supervise pixels with cross-frame instance correspondence
- **Configurable loss weights** via CLI arguments
- **Cityscapes pretrain loss** (`compute_semantic_pretrain_loss`)

### Data Augmentation (`dataset.py`)
- **Multi-scale augmentation**: random scale 0.5x–2.0x + random crop (official DeepLab2 style)
- Random horizontal flip (synchronized across frame pair + panoptic maps)
- Simplified 4-tuple output; target generation deferred to training loop

### Post-Processing (`official_postprocess.py`) — NEW
- **GPU-accelerated** panoptic decoding aligned with DeepLab2
- NMS center detection → offset-based pixel assignment → semantic-instance merge
- Gaussian heatmap rendering with local window optimization
- Greedy motion-offset tracking (assign instances to previous tracks)

### Evaluation (`run_step_eval.py`, `stq_metric.py`) — NEW
- Full STQ/AQ/IoU evaluation with bit-shift encoding (DeepLab2-aligned)
- Two post-processing backends: `--postprocess official` / `--postprocess legacy`
- GPU-accelerated inference and decoding

### Tracking (`tracking.py`) — NEW
- IoU-based Hungarian matching tracker (optimal assignment)
- Motion-projected center gating + center distance gating
- Batch mask IoU via matrix multiplication

### Cityscapes Pretraining — NEW
- `train_cityscapes.py`: full-branch pretraining (semantic + instance + offset)
- `dataset_cityscapes.py`: Cityscapes panoptic data loading
- `cityscapes_labels.py`: label ID → train ID mapping

### Training (`train.py`)
- CLI-based with full argparse (40+ configurable parameters)
- Poly / cosine / none LR schedulers
- Gradient accumulation + mixed precision + gradient clipping

### Visualization (`eval.py`)
- 2×2 grid: Input Frame | Semantic Overlay | Instance Center Heatmap | Motion Vectors (HSV)
- Proper prev_heatmap propagation via post-processed Gaussian rendering
- Generates PNG + MP4 per sequence

## File Structure

```
├── model.py                  # MotionDeepLab (ResNet-50 + dual decoder + aux head)
├── dataset.py                # KITTI-STEP dataset with multi-scale augmentation
├── dataset_cityscapes.py     # Cityscapes dataset (panoptic mode)
├── cityscapes_labels.py      # Label ID mapping
├── loss.py                   # Loss functions + target generation
├── train.py                  # KITTI-STEP training
├── train_cityscapes.py       # Cityscapes pretraining
├── official_postprocess.py   # GPU-accelerated panoptic post-processing
├── tracking.py               # IoU-based Hungarian tracker
├── run_step_eval.py          # STQ evaluation script
├── stq_metric.py             # STQ metric implementation
├── eval.py                   # Visualization (PNG + MP4)
├── run_full_pipeline_single_job.sbatch  # End-to-end pipeline
├── run_resume_pipeline.sbatch           # Resume from checkpoint
├── run_eval_only.sbatch                 # Evaluation only
├── run_eval_iou_only.sbatch             # IoU tracker evaluation
├── run_vis_all.sbatch                   # All-sequence visualization
└── outputs/                  # Evaluation results and visualizations
```

## Quick Start

```bash
# Full pipeline (Cityscapes pretrain → KITTI train → eval → visualization)
sbatch run_full_pipeline_single_job.sbatch

# Resume training from checkpoint + eval
sbatch run_resume_pipeline.sbatch

# Evaluation only (with existing checkpoint)
sbatch run_eval_only.sbatch

# Generate visualization videos for all val sequences
sbatch run_vis_all.sbatch
```

## Requirements

Tested on Aalto Triton HPC with `module load scicomp-python-env`:
- Python 3.12, PyTorch 2.x, torchvision, numpy, scipy, PIL, matplotlib, OpenCV
