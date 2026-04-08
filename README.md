# Motion-DeepLab for KITTI-STEP — Extended Implementation

This repository extends the [original course project prototype](https://github.com/cristichitz/autonomous-visual-planning-project) with a complete training, evaluation, and visualization pipeline for panoptic video segmentation on KITTI-STEP using Motion-DeepLab (ResNet-50 backbone).

## Our Contributions

Below is a summary of everything we added or improved compared to the original repository.

### 1. Cityscapes Pretraining Pipeline (NEW)

The original repository had no pretraining support. We added a full Cityscapes pretraining pipeline to initialize the model before KITTI-STEP fine-tuning:

- **`train_cityscapes.py`**: Supports two pretraining modes — semantic-only or full-branch (semantic + instance center heatmap + instance offset). Uses Top-K OHEM and poly/cosine LR schedulers.
- **`dataset_cityscapes.py`**: Loads Cityscapes images and labels. Supports panoptic mode by extracting instance IDs from `instanceIds.png` and converting Cityscapes format (labelId × 1000 + instanceNum) to KITTI-style unique instance IDs.
- **`cityscapes_labels.py`**: Utility for mapping Cityscapes raw label IDs (0–33) to train IDs (0–18 + 255 ignore).

### 2. Improved Model Architecture (`model.py`)

- Added **res4 intermediate features** to the backbone output (originally only res2, res3, res5).
- Added **auxiliary semantic segmentation head** (`semantic_aux_head`) on res4 features (1/16 scale) for deep supervision during training, following the official Panoptic-DeepLab design.
- Refactored the model into a **stateless pure encoder-decoder**: removed the internal post-processor (`PanopticPostProcessor`) and tracker (`MotionTracker`) that were embedded in the original model's `forward()`. Post-processing is now handled externally, making the model cleaner and more modular.

### 3. Enhanced Loss Functions (`loss.py`)

- **Top-K OHEM cross-entropy** (`_topk_cross_entropy`): Only backpropagates on the hardest 20% of pixels, focusing learning on difficult regions and reducing the impact of easy-to-classify pixels.
- **Small instance weighting**: Instances smaller than 4096 pixels receive 3× higher loss weight, preventing the model from ignoring small/distant objects like pedestrians.
- **Thing-only center heatmap loss**: MSE loss is now computed only on thing-class pixels (person, rider) instead of the full image, eliminating gradient dilution from background.
- **Corrected motion offset targets** (`generate_motion_targets`): For each pixel in the current frame, the motion offset now correctly points to the center of the **same instance in the previous frame**. The original implementation had incorrect motion supervision.
- **Separate motion weights**: Motion loss is only supervised on pixels that have a valid cross-frame instance correspondence, preventing incorrect gradients from unmatched regions.
- **Configurable loss weights**: All loss coefficients (semantic, center, offset, motion, auxiliary) are exposed as CLI arguments for easy tuning.
- **Cityscapes pretrain loss** (`compute_semantic_pretrain_loss`): Dedicated loss function for Cityscapes pretraining with optional auxiliary head support.

### 4. Data Augmentation (`dataset.py`)

- **Multi-scale augmentation**: Random scale between 0.5×–2.0× followed by random crop, matching the official DeepLab2 augmentation strategy. Scale factors are discretized in 0.1 steps.
- **Random horizontal flip**: Applied with 50% probability, synchronized across the frame pair and their panoptic maps.
- **Simplified dataset interface**: Returns a uniform 4-tuple `(stacked_images, semantic, instance, prev_instance)` for all splits. Target generation (heatmaps, offsets, motion targets) is deferred to the training loop for flexibility.

### 5. GPU-Accelerated Post-Processing (`official_postprocess.py`) — NEW

Replaced the original `post_processor.py` (GPU-based but tightly coupled to the model) with a standalone, DeepLab2-aligned post-processing module:

- **NMS center detection**: Threshold + max-pooling NMS on the center heatmap to find instance centers.
- **Offset-based pixel assignment**: Each pixel is assigned to the closest detected center using the predicted offset vectors. Runs entirely on GPU using PyTorch tensors.
- **Semantic-instance panoptic merge**: Combines semantic segmentation with instance assignments, using per-class instance counting.
- **Gaussian heatmap rendering**: Renders clean Gaussian heatmaps from detected centers for the next frame's 7th input channel. Uses local window optimization (3σ radius) instead of full-image computation.
- **Motion-offset tracking**: Greedy confidence-sorted assignment of current instances to previous tracks using motion-projected center distances, following the DeepLab2 tracking strategy.

### 6. STQ Evaluation System — NEW

The original repository had evaluation embedded inside `eval.py`. We created a dedicated, modular evaluation system:

- **`run_step_eval.py`**: Standalone evaluation script that runs model inference on the full validation set and computes STQ metrics. Supports two post-processing backends via `--postprocess official` (motion-offset tracking) and `--postprocess legacy` (IoU-based Hungarian matching). All parameters (center threshold, NMS kernel, tracking sigma, etc.) are configurable via CLI.
- **`stq_metric.py`**: Pure NumPy implementation of the Segmentation and Tracking Quality (STQ) metric, closely following the [DeepLab2 reference implementation](https://github.com/google-research/deeplab2). Uses bit-shift encoding (label << 16 + instance) for numerical robustness. Computes STQ = √(AQ × IoU) per sequence and globally.

### 7. IoU-Based Hungarian Tracker (`tracking.py`) — NEW

An alternative tracker using optimal matching instead of the greedy approach:

- **Batch mask IoU**: Computes the full IoU matrix between all current instances and previous tracks using matrix multiplication (`inst_flat @ track_flat.T`), replacing the O(n²) per-pair loop.
- **Hungarian matching**: Uses `scipy.optimize.linear_sum_assignment` for globally optimal instance-to-track assignment with a weighted cost combining IoU similarity and center distance.
- **Motion gating**: Filters candidate matches using motion-projected center distance and center distance gates to avoid implausible assignments.
- **Track memory**: Maintains tracks for up to `sigma` frames without updates before pruning.

### 8. Training Improvements (`train.py`)

- **Full CLI parameterization**: 40+ configurable hyperparameters via argparse (learning rate, batch size, loss weights, augmentation, scheduler, etc.).
- **LR schedulers**: Supports polynomial decay, cosine annealing, or fixed learning rate via `--lr_schedule`.
- **Gradient accumulation**: Configurable accumulation steps for effective larger batch sizes on limited GPU memory.
- **Mixed precision training**: AMP (autocast + GradScaler) for faster training and lower memory usage.
- **Gradient clipping**: Max norm = 10.0 to prevent gradient explosions.

### 9. Visualization System (`eval.py`)

Completely rewritten from the original (which produced a simple 1×2 grid with embedded STQ evaluation):

- **2×2 visualization grid**: Input Frame | Semantic Overlay (Cityscapes palette) | Instance Center Heatmap (magma colormap) | Motion Vectors (HSV-encoded direction and magnitude).
- **Pure OpenCV rendering**: Replaced matplotlib with direct OpenCV operations for ~10× faster frame generation.
- **Correct temporal propagation**: The previous-frame heatmap (`prev_heatmap`) is generated through proper post-processing (NMS → Gaussian rendering), not raw sigmoid output. This prevents error accumulation across frames.
- **Per-sequence output**: Generates `evaluation_video_{seq}.mp4` and `evaluation_result_{seq}.png` for each sequence.

### 10. Slurm Automation Scripts — NEW

| Script | Purpose |
|--------|---------|
| `run_full_pipeline_single_job.sbatch` | End-to-end: Cityscapes pretrain → KITTI train → STQ eval → visualization |
| `run_resume_pipeline.sbatch` | Resume training from a checkpoint + eval + visualization |
| `run_eval_only.sbatch` | Run full STQ evaluation + visualization on an existing checkpoint |
| `run_vis_all.sbatch` | Generate visualization videos for all 9 validation sequences |

## Results

### STQ Evaluation (epoch 180, full validation set — 9 sequences, 2981 frames)

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

### Improvement Over Original Baseline

| Version | STQ | AQ | IoU |
|---------|------|------|------|
| **Ours (epoch 180)** | **0.477** | **0.429** | **0.531** |
| **Improvement** | **+47%** | **+87%** | **+16%** |

## Training Pipeline

```
Stage 1: Cityscapes Pretraining (train_cityscapes.py)
   ├── Full-branch: semantic + instance center + offset
   ├── Top-K OHEM cross-entropy (hardest 20% pixels)
   └── ~50 epochs on Cityscapes train set

Stage 2: KITTI-STEP Fine-tuning (train.py)
   ├── Load Cityscapes pretrained weights
   ├── Multi-scale augmentation (0.5x–2.0x random scale + crop + flip)
   ├── Top-K semantic loss + small instance weighting (3x for <4096px)
   ├── Correct motion offset targets (current→previous instance centers)
   ├── Deep supervision via res4 auxiliary semantic head
   ├── Mixed precision (AMP) + gradient accumulation
   └── ~180 epochs

Stage 3: STQ Evaluation (run_step_eval.py)
   ├── GPU-accelerated post-processing
   ├── Motion-offset tracking (DeepLab2-aligned)
   └── STQ / AQ / IoU metrics

Stage 4: Visualization (eval.py)
   ├── 2×2 grid per frame (OpenCV)
   └── PNG preview + MP4 video per sequence
```

## File Structure

```
├── model.py                  # MotionDeepLab (ResNet-50 + dual decoder + aux head)
├── dataset.py                # KITTI-STEP dataset with multi-scale augmentation
├── dataset_cityscapes.py     # Cityscapes dataset (panoptic mode)
├── cityscapes_labels.py      # Label ID mapping
├── loss.py                   # Loss functions + target generation
├── train.py                  # KITTI-STEP training script
├── train_cityscapes.py       # Cityscapes pretraining script
├── official_postprocess.py   # GPU-accelerated panoptic post-processing
├── tracking.py               # IoU-based Hungarian tracker
├── run_step_eval.py          # STQ evaluation script
├── stq_metric.py             # STQ metric (DeepLab2-aligned)
├── eval.py                   # Visualization (PNG + MP4)
├── run_full_pipeline_single_job.sbatch  # End-to-end Slurm pipeline
├── run_resume_pipeline.sbatch           # Resume training + eval
├── run_eval_only.sbatch                 # Evaluation only
├── run_vis_all.sbatch                   # All-sequence visualization
└── outputs/                             # Results and visualizations
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
- Python 3.12, PyTorch 2.x, torchvision, numpy, scipy, PIL, OpenCV
