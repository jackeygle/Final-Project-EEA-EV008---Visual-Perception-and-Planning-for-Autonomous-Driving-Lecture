"""Panoptic + Motion-DeepLab style post-processing (aligned with DeepLab2).

GPU-accelerated version: heavy pixel-level ops run on CUDA when available.

References (Apache-2.0):
- deeplab2/model/post_processor/panoptic_deeplab.py
- deeplab2/model/post_processor/motion_deeplab.py
- deeplab2/video/motion_deeplab.py
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _centers_from_heatmap_nms(
    heatmap_logits: torch.Tensor,
    center_threshold: float,
    nms_kernel: int,
    keep_k_centers: int,
) -> torch.Tensor:
    """Threshold + NMS on GPU; returns (N, 2) int tensor of (y, x) centers."""
    heat = torch.sigmoid(heatmap_logits[0])
    heat = torch.where(heat > center_threshold, heat, torch.zeros_like(heat))
    pad = nms_kernel // 2
    pooled = F.max_pool2d(heat.unsqueeze(0).unsqueeze(0),
                          kernel_size=nms_kernel, stride=1, padding=pad)[0, 0]
    keep = (pooled == heat) & (heat > 0)
    ys, xs = torch.where(keep)
    if ys.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=heat.device), heat

    if keep_k_centers > 0 and ys.numel() > keep_k_centers:
        scores = heat[ys, xs]
        topk_idx = torch.topk(scores, k=keep_k_centers).indices
        ys, xs = ys[topk_idx], xs[topk_idx]
        mask = torch.zeros_like(heat)
        mask[ys, xs] = heat[ys, xs]
        heat = mask

    centers = torch.stack([ys, xs], dim=1)
    return centers, heat


def _closest_center_per_pixel_gpu(
    centers_yx: torch.Tensor,
    center_offsets_yx: torch.Tensor,
) -> torch.Tensor:
    """GPU argmin: assign each pixel to closest center via offsets. Returns (H,W) int."""
    h, w = center_offsets_yx.shape[1], center_offsets_yx.shape[2]
    n = centers_yx.shape[0]
    if n == 0:
        return torch.zeros((h, w), dtype=torch.long, device=centers_yx.device)

    yy = torch.arange(h, dtype=torch.float32, device=centers_yx.device).view(-1, 1).expand(h, w)
    xx = torch.arange(w, dtype=torch.float32, device=centers_yx.device).view(1, -1).expand(h, w)
    cy = yy + center_offsets_yx[0]
    cx = xx + center_offsets_yx[1]

    coords = torch.stack([cy, cx], dim=-1)
    cc = centers_yx.float()
    d2 = torch.sum((coords.unsqueeze(2) - cc.view(1, 1, n, 2)) ** 2, dim=-1)
    idx = torch.argmin(d2, dim=2)
    return idx


def merge_semantic_instance_panoptic_gpu(
    semantic: torch.Tensor,
    instance_map: torch.Tensor,
    thing_class_ids: List[int],
    label_divisor: int,
    void_label: int,
    stuff_area_limit: int,
) -> torch.Tensor:
    """GPU merge semantic + instance → panoptic IDs."""
    h, w = semantic.shape
    pan = torch.full((h, w), void_label * label_divisor, dtype=torch.long, device=semantic.device)

    thing_mask = torch.zeros((h, w), dtype=torch.bool, device=semantic.device)
    for c in thing_class_ids:
        thing_mask |= semantic == c

    thing_set = set(thing_class_ids)
    num_instance_per_sem = {c: 0 for c in thing_class_ids}

    inst_ids = torch.unique(instance_map)
    for iid in inst_ids:
        if iid == 0:
            continue
        imask = (instance_map == iid) & thing_mask
        if not imask.any():
            continue
        vals = semantic[imask]
        sem_major = int(torch.bincount(vals.long()).argmax())
        if sem_major not in thing_set:
            continue
        num_instance_per_sem[sem_major] += 1
        new_iid = num_instance_per_sem[sem_major]
        pan[imask] = sem_major * label_divisor + new_iid

    no_inst = instance_map == 0
    for sem_id in torch.unique(semantic):
        sem_id = int(sem_id)
        if sem_id in thing_set:
            continue
        sm = (semantic == sem_id) & no_inst
        area = int(sm.sum())
        if stuff_area_limit > 0 and area < stuff_area_limit:
            continue
        if area > 0:
            pan[sm] = sem_id * label_divisor

    return pan


def render_panoptic_gaussian_heatmap_gpu(
    panoptic_map: torch.Tensor,
    sigma: int,
    label_divisor: int,
    void_label: int,
) -> torch.Tensor:
    """GPU Gaussian heatmap — local window per instance center."""
    device = panoptic_map.device
    h, w = panoptic_map.shape
    out = torch.zeros((h, w), dtype=torch.float32, device=device)
    radius = int(3.0 * sigma)

    for pan_id in torch.unique(panoptic_map):
        pan_id = int(pan_id)
        sem_id = pan_id // label_divisor
        if sem_id == void_label or (pan_id % label_divisor) == 0:
            continue
        ys, xs = torch.where(panoptic_map == pan_id)
        if ys.numel() == 0:
            continue
        cy = float(ys.float().mean().round())
        cx = float(xs.float().mean().round())
        y0 = max(int(cy) - radius, 0)
        y1 = min(int(cy) + radius + 1, h)
        x0 = max(int(cx) - radius, 0)
        x1 = min(int(cx) + radius + 1, w)
        yy = torch.arange(y0, y1, dtype=torch.float32, device=device).view(-1, 1)
        xx = torch.arange(x0, x1, dtype=torch.float32, device=device).view(1, -1)
        g = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
        torch.maximum(out[y0:y1, x0:x1], g, out=out[y0:y1, x0:x1])
    return out


def extract_centers_from_panoptic_gpu(
    panoptic_map: torch.Tensor,
    label_divisor: int,
    void_label: int,
) -> np.ndarray:
    """Extract [x, y, panoptic_id, mask_radius, 0] rows. Returns numpy on CPU."""
    rows = []
    for pid in torch.unique(panoptic_map):
        pid = int(pid)
        sem_id = pid // label_divisor
        if sem_id == void_label or (pid % label_divisor) == 0:
            continue
        ys, xs = torch.where(panoptic_map == pid)
        if ys.numel() == 0:
            continue
        dy = int(ys.max() - ys.min() + 1)
        dx = int(xs.max() - xs.min() + 1)
        mask_radius = dy * dx
        cx = int(xs.float().mean().round())
        cy = int(ys.float().mean().round())
        rows.append([cx, cy, pid, mask_radius, 0])
    if not rows:
        return np.zeros((0, 5), dtype=np.int32)
    return np.array(rows, dtype=np.int32)


def decode_panoptic_official(
    semantic_logits: torch.Tensor,
    center_heatmap_logits: torch.Tensor,
    center_offsets_yx: torch.Tensor,
    thing_class_ids: List[int],
    label_divisor: int,
    void_label: int,
    center_threshold: float = 0.1,
    nms_kernel: int = 13,
    keep_k_centers: int = 200,
    stuff_area_limit: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU-accelerated panoptic decoding.

    Returns (all numpy on CPU):
      panoptic_map [H,W],
      rendered_heatmap [H,W] for next-frame 7th channel,
      current_centers [Nc,5].
    """
    device = semantic_logits.device
    sem = torch.argmax(semantic_logits, dim=0)

    thing_mask = torch.zeros_like(sem, dtype=torch.bool)
    for c in thing_class_ids:
        thing_mask |= sem == c

    centers_yx, _heat_nms = _centers_from_heatmap_nms(
        center_heatmap_logits, center_threshold, nms_kernel, keep_k_centers
    )

    if centers_yx.shape[0] == 0:
        pan = merge_semantic_instance_panoptic_gpu(
            sem,
            torch.zeros_like(sem, dtype=torch.long),
            thing_class_ids, label_divisor, void_label, stuff_area_limit,
        )
        rendered = render_panoptic_gaussian_heatmap_gpu(
            pan, sigma=8, label_divisor=label_divisor, void_label=void_label
        )
        return pan.cpu().numpy(), rendered.cpu().numpy(), np.zeros((0, 5), dtype=np.int32)

    inst_idx = _closest_center_per_pixel_gpu(centers_yx, center_offsets_yx)
    instance_map = torch.where(thing_mask, (inst_idx + 1).long(), torch.zeros_like(inst_idx))

    pan = merge_semantic_instance_panoptic_gpu(
        sem, instance_map, thing_class_ids, label_divisor, void_label, stuff_area_limit
    )
    rendered = render_panoptic_gaussian_heatmap_gpu(
        pan, sigma=8, label_divisor=label_divisor, void_label=void_label
    )
    centers_np = extract_centers_from_panoptic_gpu(pan, label_divisor, void_label)
    return pan.cpu().numpy(), rendered.cpu().numpy(), centers_np


def assign_instances_to_previous_tracks_numpy(
    prev_centers: np.ndarray,
    current_centers: np.ndarray,
    heatmap_hw: np.ndarray,
    motion_offsets_yx: np.ndarray,
    panoptic_map: np.ndarray,
    next_id: int,
    label_divisor: int,
    sigma: int = 7,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Greedy assignment (confidence order) matching motion_deeplab.assign_instances_to_previous_tracks."""
    h, w = panoptic_map.shape
    pan = panoptic_map.copy()
    if current_centers.shape[0] == 0:
        return pan, _increment_inactivity_and_prune(prev_centers, sigma), next_id

    confidences = []
    for row in range(current_centers.shape[0]):
        cx, cy = int(current_centers[row, 0]), int(current_centers[row, 1])
        if 0 <= cy < h and 0 <= cx < w:
            confidences.append(float(heatmap_hw[cy, cx]))
        else:
            confidences.append(0.0)
    order = np.argsort(-np.asarray(confidences))

    cur = current_centers.astype(np.int64).copy()
    prev = prev_centers.astype(np.float64).copy() if prev_centers.size else np.zeros((0, 5), dtype=np.float64)

    for ii in order:
        row = int(ii)
        center_id = int(cur[row, 2])
        cx_i, cy_i = int(cur[row, 0]), int(cur[row, 1])
        oyx = motion_offsets_yx[:, cy_i, cx_i].astype(np.float64)
        oxy = np.array([oyx[1], oyx[0]], dtype=np.float64)
        center_loc = oxy + np.array([float(cx_i), float(cy_i)], dtype=np.float64)

        mask = pan == center_id
        if not np.any(mask):
            continue
        center_sem = center_id // label_divisor

        if prev.shape[0] == 0:
            new_id = center_sem * label_divisor + next_id
            pan[mask] = new_id
            cur[row, 2] = new_id
            next_id += 1
            continue

        same = (prev[:, 2].astype(np.int64) // label_divisor) == center_sem
        prev_same = prev[same]
        if prev_same.shape[0] == 0:
            new_id = center_sem * label_divisor + next_id
            pan[mask] = new_id
            cur[row, 2] = new_id
            next_id += 1
            continue

        d2 = np.sum((prev_same[:, :2].astype(np.float64) - center_loc[None, :]) ** 2, axis=1)
        j = int(np.argmin(d2))
        rad = float(prev_same[j, 3])
        if d2[j] < rad:
            new_center_id = int(prev_same[j, 2])
            pan[mask] = new_center_id
            cur[row, 2] = new_center_id
            keep = prev[:, 2].astype(np.int64) != new_center_id
            prev = prev[keep]
        else:
            new_id = center_sem * label_divisor + next_id
            pan[mask] = new_id
            cur[row, 2] = new_id
            next_id += 1

    if prev.shape[0] > 0:
        cur = np.vstack([cur, prev]) if cur.size else prev

    if cur.shape[0] > 0:
        cur = cur.copy()
        cur[:, 4] = cur[:, 4] + 1
        cur = cur[cur[:, 4] <= sigma]

    return pan, cur.astype(np.int32), next_id


def _increment_inactivity_and_prune(prev_centers: np.ndarray, sigma: int) -> np.ndarray:
    if prev_centers.size == 0:
        return prev_centers.astype(np.int32)
    prev_centers = prev_centers.astype(np.float64).copy()
    prev_centers[:, 4] += 1
    prev_centers = prev_centers[prev_centers[:, 4] <= sigma]
    return prev_centers.astype(np.int32)
