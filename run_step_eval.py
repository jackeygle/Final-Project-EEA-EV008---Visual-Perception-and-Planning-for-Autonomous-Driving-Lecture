"""Run KITTI-STEP-style evaluation (STQ/AQ/IoU) for MotionDeepLab.

Use ``--postprocess official`` (default) for DeepLab2-aligned decoding + tracking.
Use ``--postprocess legacy`` for the earlier IoU-tracker baseline.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from dataset import KittiStepDataset
from model import MotionDeepLab
from official_postprocess import (
    assign_instances_to_previous_tracks_numpy,
    decode_panoptic_official,
)
from stq_metric import STQuality
from tracking import IoUTracker


def _extract_centers(center_heatmap: np.ndarray, threshold: float, k: int) -> np.ndarray:
    """Legacy: top-k local maxima as centers [[y, x], ...]."""
    h, w = center_heatmap.shape
    heat = torch.from_numpy(center_heatmap).unsqueeze(0).unsqueeze(0).float()
    pooled = F.max_pool2d(heat, kernel_size=7, stride=1, padding=3)
    keep = (heat == pooled) & (heat >= threshold)
    ys, xs = torch.where(keep[0, 0])
    if ys.numel() == 0:
        return np.zeros((0, 2), dtype=np.int32)

    scores = heat[0, 0, ys, xs]
    topk = min(k, ys.numel())
    idx = torch.topk(scores, k=topk).indices
    centers = torch.stack([ys[idx], xs[idx]], dim=1).cpu().numpy().astype(np.int32)
    valid = np.logical_and.reduce(
        [
            centers[:, 0] >= 0,
            centers[:, 0] < h,
            centers[:, 1] >= 0,
            centers[:, 1] < w,
        ]
    )
    return centers[valid]


def _decode_panoptic_legacy(
    semantic_logits: torch.Tensor,
    center_heatmap_logits: torch.Tensor,
    center_offsets: torch.Tensor,
    thing_classes: List[int],
    label_divisor: int,
    center_threshold: float,
    max_centers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated legacy panoptic decode."""
    device = semantic_logits.device
    sem = torch.argmax(semantic_logits, dim=0)
    heat = torch.sigmoid(center_heatmap_logits[0])

    h, w = sem.shape
    inst = torch.zeros((h, w), dtype=torch.long, device=device)

    heat_np = heat.cpu().numpy().astype(np.float32)
    centers = _extract_centers(heat_np, threshold=center_threshold, k=max_centers)
    if centers.shape[0] == 0:
        return (sem * label_divisor).cpu().numpy().astype(np.int32), heat_np

    sem_np = sem.cpu().numpy()
    centers_t = torch.from_numpy(centers).to(device)

    yy = torch.arange(h, dtype=torch.float32, device=device).view(-1, 1).expand(h, w)
    xx = torch.arange(w, dtype=torch.float32, device=device).view(1, -1).expand(h, w)
    pred_cy = yy + center_offsets[0]
    pred_cx = xx + center_offsets[1]

    for cls in thing_classes:
        cls_mask = sem == cls
        if not cls_mask.any():
            continue

        cls_center_mask = sem_np[centers[:, 0], centers[:, 1]] == cls
        cls_centers = centers_t[cls_center_mask]
        if cls_centers.shape[0] == 0:
            continue

        cy_flat = pred_cy[cls_mask]
        cx_flat = pred_cx[cls_mask]
        cc_y = cls_centers[:, 0].float()
        cc_x = cls_centers[:, 1].float()
        d2 = (cy_flat.unsqueeze(1) - cc_y.unsqueeze(0)) ** 2 + \
             (cx_flat.unsqueeze(1) - cc_x.unsqueeze(0)) ** 2
        nearest = torch.argmin(d2, dim=1) + 1
        inst[cls_mask] = nearest

    panoptic = (sem * label_divisor + inst).cpu().numpy().astype(np.int32)
    return panoptic, heat_np


THING_CLASSES_KITTI_STEP = [11, 13]


def evaluate(args: argparse.Namespace) -> Dict:
    dataset = KittiStepDataset(root_dir=args.data_root, split=args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MotionDeepLab().to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    stq = STQuality(
        num_classes=19,
        things_list=THING_CLASSES_KITTI_STEP,
        ignore_label=255,
        label_bit_shift=16,
        offset=256 * 256 * 256,
    )

    if args.postprocess == "legacy":
        tracker = IoUTracker(
            classes_to_track=THING_CLASSES_KITTI_STEP,
            label_divisor=args.label_divisor,
        )
    else:
        tracker = None

    prev_seq = None
    prev_heatmap: torch.Tensor | None = None
    prev_centers = np.zeros((0, 5), dtype=np.int32)
    next_id = 1
    processed = 0
    total = len(dataset) if args.max_frames <= 0 else min(len(dataset), args.max_frames)

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample_meta = dataset.samples[idx]
            seq_id = sample_meta["sequence_id"]
            if prev_seq != seq_id:
                if args.postprocess == "legacy":
                    tracker.reset_states()
                prev_heatmap = None
                prev_centers = np.zeros((0, 5), dtype=np.int32)
                next_id = 1
                prev_seq = seq_id

            stacked_images, gt_sem, gt_inst, _ = dataset[idx]
            image = stacked_images.unsqueeze(0).to(device)

            if prev_heatmap is None:
                h, w = gt_sem.shape
                prev_heatmap = torch.zeros((1, 1, h, w), device=device)

            model_input = torch.cat([image, prev_heatmap], dim=1)

            autocast_enabled = device.type == "cuda"
            with autocast(device_type=device.type, enabled=autocast_enabled):
                pred = model(model_input)

            if args.postprocess == "official":
                pan_untracked, rendered_hw, current_centers = decode_panoptic_official(
                    pred["semantic_logits"][0],
                    pred["center_heatmap"][0],
                    pred["center_offsets"][0],
                    thing_class_ids=THING_CLASSES_KITTI_STEP,
                    label_divisor=args.label_divisor,
                    void_label=255,
                    center_threshold=args.center_threshold,
                    nms_kernel=args.nms_kernel,
                    keep_k_centers=args.max_centers,
                    stuff_area_limit=args.stuff_area_limit,
                )
                motion_yx = pred["motion_offsets"][0].detach().cpu().numpy().astype(np.float32)
                tracked_panoptic, prev_centers, next_id = assign_instances_to_previous_tracks_numpy(
                    prev_centers,
                    current_centers,
                    rendered_hw,
                    motion_yx,
                    pan_untracked,
                    next_id,
                    args.label_divisor,
                    sigma=args.track_sigma,
                )
                prev_heatmap = torch.from_numpy(rendered_hw).unsqueeze(0).unsqueeze(0).to(device)
            else:
                frame_panoptic, current_heat = _decode_panoptic_legacy(
                    pred["semantic_logits"][0],
                    pred["center_heatmap"][0],
                    pred["center_offsets"][0],
                    thing_classes=THING_CLASSES_KITTI_STEP,
                    label_divisor=args.label_divisor,
                    center_threshold=args.center_threshold,
                    max_centers=args.max_centers,
                )
                motion_yx = pred["motion_offsets"][0].detach().cpu().numpy().astype(np.float32)
                tracked_panoptic = tracker.update(frame_panoptic, motion_yx=motion_yx)
                prev_heatmap = torch.from_numpy(current_heat).unsqueeze(0).unsqueeze(0).to(device)

            gt_sem_np = gt_sem.numpy().astype(np.int64)
            gt_inst_np = gt_inst.numpy().astype(np.int64)
            pred_sem_np = (tracked_panoptic // args.label_divisor).astype(np.int64)
            pred_track_np = (tracked_panoptic % args.label_divisor).astype(np.int64)

            y_true = (gt_sem_np << 16) + gt_inst_np
            y_pred = (pred_sem_np << 16) + pred_track_np
            stq.update_state(y_true=y_true, y_pred=y_pred, sequence_id=seq_id)

            processed += 1
            if args.log_every > 0 and processed % args.log_every == 0:
                print(f"Processed {processed}/{total} frames...")
            if args.max_frames > 0 and processed >= args.max_frames:
                break

    return dict(stq.result())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--label_divisor",
        type=int,
        default=1000,
        help="KITTI-STEP / DeepLab2 uses 1000 (not 10000).",
    )
    parser.add_argument("--center_threshold", type=float, default=0.1)
    parser.add_argument("--max_centers", type=int, default=200)
    parser.add_argument("--nms_kernel", type=int, default=13)
    parser.add_argument("--stuff_area_limit", type=int, default=0)
    parser.add_argument("--track_sigma", type=int, default=7)
    parser.add_argument(
        "--postprocess",
        type=str,
        default="official",
        choices=("official", "legacy"),
    )
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--output_json", type=str, default="outputs/stq_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    results = evaluate(args)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Evaluation complete")
    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
