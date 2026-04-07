"""Panoptic + Motion-DeepLab style post-processing (aligned with DeepLab2).

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
    heatmap_hw: np.ndarray,
    center_threshold: float,
    nms_kernel: int,
    keep_k_centers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Threshold + NMS on center heatmap; returns centers (N,2) as (y,x)."""
    h, w = heatmap_hw.shape
    heat = np.where(heatmap_hw > center_threshold, heatmap_hw, 0.0).astype(np.float32)
    t = torch.from_numpy(heat).view(1, 1, h, w)
    pad = nms_kernel // 2
    pooled = F.max_pool2d(t, kernel_size=nms_kernel, stride=1, padding=pad)
    pooled = pooled.numpy()[0, 0]
    heat = np.where((pooled == heat) & (heat > 0), heat, 0.0)
    ys, xs = np.where(heat > 0.0)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.int32), heat

    centers = np.stack([ys, xs], axis=1).astype(np.int32)
    if keep_k_centers > 0 and centers.shape[0] > keep_k_centers:
        scores = heat[ys, xs]
        order = np.argsort(-scores)
        order = order[:keep_k_centers]
        centers = centers[order]
        keep = np.zeros_like(heat, dtype=np.float32)
        keep[centers[:, 0], centers[:, 1]] = heat[centers[:, 0], centers[:, 1]]
        heat = keep
    return centers, heat


def _closest_center_per_pixel(
    centers_yx: np.ndarray,
    center_offsets_yx: np.ndarray,
) -> np.ndarray:
    """Argmin over centers of || (y,x)+offset - center_k || (official panoptic_deeplab)."""
    h, w = center_offsets_yx.shape[1], center_offsets_yx.shape[2]
    if centers_yx.shape[0] == 0:
        return np.zeros((h, w), dtype=np.int32)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cy = yy.astype(np.float32) + center_offsets_yx[0]
    cx = xx.astype(np.float32) + center_offsets_yx[1]
    flat = np.stack([cy.ravel(), cx.ravel()], axis=-1)

    cc = centers_yx.astype(np.float32)
    d2 = np.sum((flat[:, None, :] - cc[None, :, :]) ** 2, axis=-1)
    idx = np.argmin(d2, axis=1).reshape(h, w).astype(np.int32)
    return idx


def merge_semantic_instance_panoptic(
    semantic: np.ndarray,
    instance_map: np.ndarray,
    thing_class_ids: List[int],
    label_divisor: int,
    void_label: int,
    stuff_area_limit: int,
) -> np.ndarray:
    """Merge semantic + class-agnostic instance ids into panoptic IDs (numpy)."""
    h, w = semantic.shape
    pan = np.ones((h, w), dtype=np.int32) * (void_label * label_divisor)
    thing_set = set(thing_class_ids)
    semantic_thing = np.zeros((h, w), dtype=bool)
    for c in thing_class_ids:
        semantic_thing |= semantic == c

    num_instance_per_sem = {c: 0 for c in thing_class_ids}

    inst_ids = np.unique(instance_map)
    for iid in inst_ids:
        if iid == 0:
            continue
        thing_mask = np.logical_and(instance_map == iid, semantic_thing)
        if not np.any(thing_mask):
            continue
        vals = semantic[thing_mask]
        sem_major = int(np.bincount(vals).argmax())
        if sem_major not in thing_set:
            continue
        num_instance_per_sem[sem_major] += 1
        new_iid = num_instance_per_sem[sem_major]
        pan = np.where(thing_mask, sem_major * label_divisor + new_iid, pan)

    inst_stuff = instance_map == 0
    for sem_id in np.unique(semantic):
        if sem_id in thing_set:
            continue
        sm = np.logical_and(semantic == sem_id, inst_stuff)
        area = int(np.count_nonzero(sm))
        if stuff_area_limit > 0 and area < stuff_area_limit:
            continue
        if area > 0:
            pan = np.where(sm, sem_id * label_divisor, pan)

    return pan


def extract_centers_from_panoptic(
    panoptic_map: np.ndarray,
    label_divisor: int,
    void_label: int,
) -> np.ndarray:
    """Rows [x, y, panoptic_id, mask_radius, 0] — same info as render_panoptic_map_as_heatmap."""
    rows = []
    for pid in np.unique(panoptic_map):
        sem_id = int(pid // label_divisor)
        if sem_id == void_label or int(pid % label_divisor) == 0:
            continue
        ys, xs = np.where(panoptic_map == pid)
        if ys.size == 0:
            continue
        dy = int(ys.max() - ys.min() + 1)
        dx = int(xs.max() - xs.min() + 1)
        mask_radius = int(round(dy * dx))
        cx = int(np.round(xs.mean()))
        cy = int(np.round(ys.mean()))
        rows.append([cx, cy, int(pid), mask_radius, 0])
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
    Returns:
      panoptic_map [H,W] (decoder merge, before track),
      rendered_heatmap [H,W] for next-frame 7th channel (gaussian),
      current_centers [Nc,5] [x,y,panoptic_id,radius,0] for assign_instances_to_previous_tracks.
    """
    sem = torch.argmax(semantic_logits, dim=0).cpu().numpy().astype(np.int32)
    heat_raw = torch.sigmoid(center_heatmap_logits[0]).cpu().numpy().astype(np.float32)
    off = center_offsets_yx.cpu().numpy().astype(np.float32)

    thing_mask = np.zeros_like(sem, dtype=bool)
    for c in thing_class_ids:
        thing_mask |= sem == c

    centers_yx, _heat_nms = _centers_from_heatmap_nms(
        heat_raw, center_threshold, nms_kernel, keep_k_centers
    )
    if centers_yx.shape[0] == 0:
        pan = merge_semantic_instance_panoptic(
            sem,
            np.zeros_like(sem, dtype=np.int32),
            thing_class_ids,
            label_divisor,
            void_label,
            stuff_area_limit,
        )
        rendered = render_panoptic_gaussian_heatmap(
            pan, sigma=8, label_divisor=label_divisor, void_label=void_label
        )
        return pan, rendered, np.zeros((0, 5), dtype=np.int32)

    inst_idx = _closest_center_per_pixel(centers_yx, off)
    instance_map = np.where(thing_mask, (inst_idx + 1).astype(np.int32), 0)

    pan = merge_semantic_instance_panoptic(
        sem, instance_map, thing_class_ids, label_divisor, void_label, stuff_area_limit
    )
    rendered = render_panoptic_gaussian_heatmap(
        pan, sigma=8, label_divisor=label_divisor, void_label=void_label
    )
    centers_np = extract_centers_from_panoptic(pan, label_divisor, void_label)
    return pan, rendered, centers_np


def render_panoptic_gaussian_heatmap(
    panoptic_map: np.ndarray,
    sigma: int,
    label_divisor: int,
    void_label: int,
) -> np.ndarray:
    """Gaussian heatmap per instance center (approximation of TF render_panoptic_map_as_heatmap)."""
    h, w = panoptic_map.shape
    out = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for pan_id in np.unique(panoptic_map):
        sem_id = int(pan_id // label_divisor)
        if sem_id == void_label or int(pan_id % label_divisor) == 0:
            continue
        ys, xs = np.where(panoptic_map == pan_id)
        if ys.size == 0:
            continue
        cy = float(np.round(ys.mean()))
        cx = float(np.round(xs.mean()))
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2)).astype(np.float32)
        out = np.maximum(out, g)
    return out


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
