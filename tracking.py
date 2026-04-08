"""STEP-style tracker with Hungarian matching and motion gating."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.count_nonzero(mask_a & mask_b)
    if inter == 0:
        return 0.0
    union = np.count_nonzero(mask_a) + np.count_nonzero(mask_b) - inter
    return float(inter / max(union, 1))


def _batch_mask_iou(inst_masks: list, track_masks: list) -> np.ndarray:
    """Compute IoU matrix between all instance masks and track masks efficiently."""
    n_i, n_t = len(inst_masks), len(track_masks)
    if n_i == 0 or n_t == 0:
        return np.zeros((n_i, n_t), dtype=np.float32)
    h, w = inst_masks[0].shape
    inst_flat = np.stack([m.ravel() for m in inst_masks], axis=0).astype(np.float32)
    track_flat = np.stack([m.ravel() for m in track_masks], axis=0).astype(np.float32)
    inter = inst_flat @ track_flat.T
    inst_area = inst_flat.sum(axis=1, keepdims=True)
    track_area = track_flat.sum(axis=1, keepdims=True)
    union = inst_area + track_area.T - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


class IoUTracker:
    """Tracks thing instances by IoU + motion/center-gated Hungarian matching."""

    def __init__(
        self,
        classes_to_track: List[int],
        label_divisor: int = 10000,
        sigma: int = 10,
        iou_threshold: float = 0.3,
        center_gate_scale: float = 2.5,
        motion_gate_scale: float = 3.0,
        iou_weight: float = 0.7,
    ):
        self.classes_to_track = classes_to_track
        self.label_divisor = label_divisor
        self.sigma = sigma
        self.iou_threshold = iou_threshold
        self.center_gate_scale = center_gate_scale
        self.motion_gate_scale = motion_gate_scale
        self.iou_weight = iou_weight
        self.reset_states()

    def reset_states(self) -> None:
        self.last_mask_per_track: Dict[int, Dict[int, np.ndarray]] = {
            c: {} for c in self.classes_to_track
        }
        self.frames_since_update: Dict[int, Dict[int, int]] = {
            c: {} for c in self.classes_to_track
        }
        self.last_center_per_track: Dict[int, Dict[int, Tuple[float, float]]] = {
            c: {} for c in self.classes_to_track
        }
        self.last_area_per_track: Dict[int, Dict[int, float]] = {
            c: {} for c in self.classes_to_track
        }
        self.next_track_id = 1

    def _add_track(self, cls: int, mask: np.ndarray, center_yx: Tuple[float, float]) -> int:
        track_id = self.next_track_id
        self.last_mask_per_track[cls][track_id] = mask
        self.frames_since_update[cls][track_id] = 0
        self.last_center_per_track[cls][track_id] = center_yx
        self.last_area_per_track[cls][track_id] = float(np.count_nonzero(mask))
        self.next_track_id += 1
        return track_id

    def _age_unmatched_tracks(self, cls: int, unmatched_track_ids: List[int]) -> None:
        for track_id in unmatched_track_ids:
            self.frames_since_update[cls][track_id] += 1
            if self.frames_since_update[cls][track_id] > self.sigma:
                del self.frames_since_update[cls][track_id]
                del self.last_mask_per_track[cls][track_id]
                del self.last_center_per_track[cls][track_id]
                del self.last_area_per_track[cls][track_id]

    @staticmethod
    def _mask_center(mask: np.ndarray) -> Tuple[float, float]:
        ys, xs = np.where(mask)
        if ys.size == 0:
            return 0.0, 0.0
        return float(ys.mean()), float(xs.mean())

    @staticmethod
    def _mean_motion_prev_center(mask: np.ndarray, motion_yx: np.ndarray) -> Tuple[float, float]:
        ys, xs = np.where(mask)
        if ys.size == 0:
            return 0.0, 0.0
        py = ys.astype(np.float32) + motion_yx[0, ys, xs]
        px = xs.astype(np.float32) + motion_yx[1, ys, xs]
        return float(py.mean()), float(px.mean())

    def update(self, panoptic_frame: np.ndarray, motion_yx: np.ndarray | None = None) -> np.ndarray:
        """Assigns stable tracking IDs for thing classes in a panoptic map."""
        pred_sem = panoptic_frame // self.label_divisor
        pred_inst = panoptic_frame % self.label_divisor
        out_inst = np.zeros_like(pred_inst, dtype=np.int32)

        for cls in self.classes_to_track:
            cls_mask = np.logical_and(pred_sem == cls, pred_inst > 0)
            inst_ids = np.unique(pred_inst[cls_mask])
            inst_masks = [np.logical_and(cls_mask, pred_inst == iid) for iid in inst_ids]
            inst_centers = [self._mask_center(m) for m in inst_masks]
            inst_prev_centers = (
                [self._mean_motion_prev_center(m, motion_yx) for m in inst_masks]
                if motion_yx is not None
                else inst_centers
            )
            inst_areas = [float(np.count_nonzero(m)) for m in inst_masks]

            if len(inst_masks) == 0:
                self._age_unmatched_tracks(cls, list(self.last_mask_per_track[cls].keys()))
                continue

            track_ids = list(self.last_mask_per_track[cls].keys())
            if len(track_ids) == 0:
                for i, mask in enumerate(inst_masks):
                    new_id = self._add_track(cls, mask, inst_centers[i])
                    out_inst[mask] = new_id
                continue

            n_i, n_t = len(inst_masks), len(track_ids)
            track_masks_list = [self.last_mask_per_track[cls][tid] for tid in track_ids]
            iou_matrix = _batch_mask_iou(inst_masks, track_masks_list)

            cost_matrix = np.full((n_i, n_t), 1e6, dtype=np.float32)
            for i in range(n_i):
                cy, cx = inst_centers[i]
                py, px = inst_prev_centers[i]
                for j, tid in enumerate(track_ids):
                    ty, tx = self.last_center_per_track[cls][tid]
                    area = max(inst_areas[i], self.last_area_per_track[cls][tid], 1.0)
                    radius = np.sqrt(area)
                    center_dist = np.sqrt((cy - ty) ** 2 + (cx - tx) ** 2)
                    motion_prev_dist = np.sqrt((py - ty) ** 2 + (px - tx) ** 2)

                    center_gate = center_dist <= (self.center_gate_scale * radius)
                    motion_gate = motion_prev_dist <= (self.motion_gate_scale * radius)
                    if not (center_gate or motion_gate):
                        continue

                    dist_term = min(center_dist / (self.center_gate_scale * radius + 1e-6), 1.0)
                    cost_matrix[i, j] = (1.0 - self.iou_weight) * dist_term + self.iou_weight * (1.0 - iou_matrix[i, j])

            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            matched_inst = set()
            matched_track = set()
            for i, j in zip(row_idx.tolist(), col_idx.tolist()):
                if cost_matrix[i, j] >= 1e5:
                    continue
                if iou_matrix[i, j] < self.iou_threshold:
                    continue
                tid = track_ids[j]
                mask = inst_masks[i]
                self.last_mask_per_track[cls][tid] = mask
                self.last_center_per_track[cls][tid] = inst_centers[i]
                self.last_area_per_track[cls][tid] = inst_areas[i]
                self.frames_since_update[cls][tid] = 0
                out_inst[mask] = tid
                matched_inst.add(i)
                matched_track.add(j)

            unmatched_inst = [i for i in range(n_i) if i not in matched_inst]
            unmatched_tracks = [track_ids[j] for j in range(n_t) if j not in matched_track]

            for i in unmatched_inst:
                mask = inst_masks[i]
                new_id = self._add_track(cls, mask, inst_centers[i])
                out_inst[mask] = new_id

            self._age_unmatched_tracks(cls, unmatched_tracks)

        if self.next_track_id >= self.label_divisor:
            raise ValueError("Too many tracks for current label_divisor.")

        return pred_sem * self.label_divisor + out_inst
