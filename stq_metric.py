"""Numpy STQ metric adapted from DeepLab2's reference implementation.

Original implementation:
https://github.com/google-research/deeplab2/blob/main/evaluation/numpy/segmentation_and_tracking_quality.py
"""

from __future__ import annotations

import collections
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np

_EPSILON = 1e-15


def _update_dict_stats(
    stat_dict: MutableMapping[int, float],
    id_array: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> None:
    """Updates a dictionary with pixel counts for IDs."""
    if weights is None:
        ids, counts = np.unique(id_array, return_counts=True)
        for idx, count in zip(ids, counts.astype(np.float64)):
            stat_dict[int(idx)] = stat_dict.get(int(idx), 0.0) + float(count)
        return

    unique_weights = np.unique(weights).tolist()
    for weight in unique_weights:
        ids, counts = np.unique(id_array[weights == weight], return_counts=True)
        for idx, count in zip(ids, counts.astype(np.float64)):
            stat_dict[int(idx)] = stat_dict.get(int(idx), 0.0) + float(count * weight)


class STQuality:
    """Segmentation and Tracking Quality (STQ) for panoptic video predictions."""

    def __init__(
        self,
        num_classes: int,
        things_list: Sequence[int],
        ignore_label: int,
        label_bit_shift: int,
        offset: int,
    ):
        self._num_classes = num_classes
        self._things_list = list(things_list)
        self._ignore_label = ignore_label
        self._label_bit_shift = label_bit_shift
        self._bit_mask = (2**label_bit_shift) - 1
        self._offset = offset

        if ignore_label >= num_classes:
            self._confusion_matrix_size = num_classes + 1
            self._include_indices = np.arange(self._num_classes)
        else:
            self._confusion_matrix_size = num_classes
            self._include_indices = np.array(
                [i for i in range(num_classes) if i != self._ignore_label]
            )

        lower_bound = num_classes << label_bit_shift
        if offset < lower_bound:
            raise ValueError(
                f"offset={offset} is too small; use >= {lower_bound} "
                "to avoid collisions."
            )

        self.reset_states()

    def reset_states(self) -> None:
        self._iou_confusion_matrix_per_sequence = collections.OrderedDict()
        self._predictions = collections.OrderedDict()
        self._ground_truth = collections.OrderedDict()
        self._intersections = collections.OrderedDict()
        self._sequence_length = collections.OrderedDict()

    def get_semantic(self, y: np.ndarray) -> np.ndarray:
        return y >> self._label_bit_shift

    def _get_or_update_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: Optional[np.ndarray] = None,
        confusion_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if confusion_matrix is None:
            confusion_matrix = np.zeros(
                (self._confusion_matrix_size, self._confusion_matrix_size),
                dtype=np.float64,
            )

        if weights is None:
            idxs = (np.reshape(y_true, [-1]) << self._label_bit_shift) + np.reshape(
                y_pred, [-1]
            )
            unique_idxs, counts = np.unique(idxs, return_counts=True)
            confusion_matrix[
                self.get_semantic(unique_idxs), unique_idxs & self._bit_mask
            ] += counts.astype(np.float64)
            return confusion_matrix

        weights = np.reshape(weights, [-1])
        idxs = (np.reshape(y_true, [-1]) << self._label_bit_shift) + np.reshape(
            y_pred, [-1]
        )
        unique_weights = np.unique(weights).tolist()
        for weight in unique_weights:
            idxs_masked = idxs[weights == weight]
            unique_idxs, counts = np.unique(idxs_masked, return_counts=True)
            confusion_matrix[
                self.get_semantic(unique_idxs), unique_idxs & self._bit_mask
            ] += counts.astype(np.float64) * weight
        return confusion_matrix

    def update_state(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sequence_id=0,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        if weights is not None:
            weights = weights.reshape(y_true.shape)

        semantic_label = self.get_semantic(y_true)
        semantic_prediction = self.get_semantic(y_pred)

        if self._ignore_label > self._num_classes:
            semantic_label = np.where(
                semantic_label != self._ignore_label, semantic_label, self._num_classes
            )
            semantic_prediction = np.where(
                semantic_prediction != self._ignore_label,
                semantic_prediction,
                self._num_classes,
            )

        max_ci = self._confusion_matrix_size - 1
        semantic_label = np.clip(semantic_label, 0, max_ci)
        semantic_prediction = np.clip(semantic_prediction, 0, max_ci)

        if sequence_id in self._iou_confusion_matrix_per_sequence:
            self._iou_confusion_matrix_per_sequence[
                sequence_id
            ] = self._get_or_update_confusion_matrix(
                semantic_label,
                semantic_prediction,
                weights,
                self._iou_confusion_matrix_per_sequence[sequence_id],
            )
            self._sequence_length[sequence_id] += 1
        else:
            self._iou_confusion_matrix_per_sequence[
                sequence_id
            ] = self._get_or_update_confusion_matrix(
                semantic_label, semantic_prediction, weights, None
            )
            self._predictions[sequence_id] = {}
            self._ground_truth[sequence_id] = {}
            self._intersections[sequence_id] = {}
            self._sequence_length[sequence_id] = 1

        instance_label = y_true & self._bit_mask

        label_mask = np.zeros_like(semantic_label, dtype=bool)
        prediction_mask = np.zeros_like(semantic_prediction, dtype=bool)
        for thing_cls in self._things_list:
            label_mask |= semantic_label == thing_cls
            prediction_mask |= semantic_prediction == thing_cls

        is_crowd = np.logical_and(instance_label == 0, label_mask)
        label_mask = np.logical_and(label_mask, np.logical_not(is_crowd))
        prediction_mask = np.logical_and(prediction_mask, np.logical_not(is_crowd))

        seq_preds = self._predictions[sequence_id]
        seq_gts = self._ground_truth[sequence_id]
        seq_intersects = self._intersections[sequence_id]

        _update_dict_stats(
            seq_preds,
            y_pred[prediction_mask],
            weights[prediction_mask] if weights is not None else None,
        )
        _update_dict_stats(
            seq_gts,
            y_true[label_mask],
            weights[label_mask] if weights is not None else None,
        )

        non_crowd_intersection = np.logical_and(label_mask, prediction_mask)
        intersection_ids = (
            y_true[non_crowd_intersection] * self._offset
            + y_pred[non_crowd_intersection]
        )
        _update_dict_stats(
            seq_intersects,
            intersection_ids,
            weights[non_crowd_intersection] if weights is not None else None,
        )

    def result(self) -> Mapping[str, Any]:
        num_tubes_per_seq = [0] * len(self._ground_truth)
        aq_per_seq = [0.0] * len(self._ground_truth)
        iou_per_seq = [0.0] * len(self._ground_truth)
        id_per_seq = [""] * len(self._ground_truth)

        for index, sequence_id in enumerate(self._ground_truth):
            outer_sum = 0.0
            predictions = self._predictions[sequence_id]
            ground_truth = self._ground_truth[sequence_id]
            intersections = self._intersections[sequence_id]
            num_tubes_per_seq[index] = len(ground_truth)
            id_per_seq[index] = str(sequence_id)

            for gt_id, gt_size in ground_truth.items():
                inner_sum = 0.0
                for pr_id, pr_size in predictions.items():
                    tpa_key = self._offset * gt_id + pr_id
                    if tpa_key in intersections:
                        tpa = intersections[tpa_key]
                        fpa = pr_size - tpa
                        fna = gt_size - tpa
                        inner_sum += tpa * (tpa / (tpa + fpa + fna))
                outer_sum += (1.0 / gt_size) * inner_sum
            aq_per_seq[index] = outer_sum

        aq_per_seq = np.asarray(aq_per_seq, dtype=np.float64)
        num_tubes_per_seq = np.asarray(num_tubes_per_seq, dtype=np.float64)
        aq_mean = np.sum(aq_per_seq) / np.maximum(np.sum(num_tubes_per_seq), _EPSILON)
        aq_per_seq = aq_per_seq / np.maximum(num_tubes_per_seq, _EPSILON)

        total_confusion = np.zeros(
            (self._confusion_matrix_size, self._confusion_matrix_size), dtype=np.float64
        )
        for index, confusion in enumerate(self._iou_confusion_matrix_per_sequence.values()):
            removal_matrix = np.zeros_like(confusion)
            removal_matrix[self._include_indices, :] = 1.0
            confusion = confusion * removal_matrix
            total_confusion += confusion

            intersections = confusion.diagonal()
            fps = confusion.sum(axis=0) - intersections
            fns = confusion.sum(axis=1) - intersections
            unions = intersections + fps + fns
            num_classes = np.count_nonzero(unions)
            ious = intersections / np.maximum(unions, _EPSILON)
            iou_per_seq[index] = float(np.sum(ious) / max(num_classes, 1))

        intersections = total_confusion.diagonal()
        fps = total_confusion.sum(axis=0) - intersections
        fns = total_confusion.sum(axis=1) - intersections
        unions = intersections + fps + fns
        num_classes = np.count_nonzero(unions)
        ious = intersections / np.maximum(unions, _EPSILON)
        iou_mean = float(np.sum(ious) / max(num_classes, 1))

        stq = float(np.sqrt(aq_mean * iou_mean))
        stq_per_seq = np.sqrt(aq_per_seq * np.asarray(iou_per_seq))

        return {
            "STQ": stq,
            "AQ": float(aq_mean),
            "IoU": iou_mean,
            "STQ_per_seq": stq_per_seq.tolist(),
            "AQ_per_seq": aq_per_seq.tolist(),
            "IoU_per_seq": iou_per_seq,
            "ID_per_seq": id_per_seq,
            "Length_per_seq": list(self._sequence_length.values()),
        }
