"""
Cityscapes labelId -> trainId (19 classes + ignore 255), from official
cityscapesscripts/helpers/labels.py — used when only *_gtFine_labelIds.png exists.
"""
import numpy as np

# (label_id, train_id) for ids 0..33; omitted ids default to 255 in LUT
_ID_TRAIN_PAIRS = (
    (0, 255),
    (1, 255),
    (2, 255),
    (3, 255),
    (4, 255),
    (5, 255),
    (6, 255),
    (7, 0),
    (8, 1),
    (9, 255),
    (10, 255),
    (11, 2),
    (12, 3),
    (13, 4),
    (14, 255),
    (15, 255),
    (16, 255),
    (17, 5),
    (18, 255),
    (19, 6),
    (20, 7),
    (21, 8),
    (22, 9),
    (23, 10),
    (24, 11),
    (25, 12),
    (26, 13),
    (27, 14),
    (28, 15),
    (29, 255),
    (30, 255),
    (31, 16),
    (32, 17),
    (33, 18),
)

ID_TO_TRAINID_LUT = np.full(256, 255, dtype=np.uint8)
for lid, tid in _ID_TRAIN_PAIRS:
    ID_TO_TRAINID_LUT[lid] = tid


def labelids_to_trainids(arr: np.ndarray) -> np.ndarray:
    """Map Cityscapes labelIds image to trainIds (HxW)."""
    a = np.asarray(arr)
    return ID_TO_TRAINID_LUT[np.clip(a, 0, 255).astype(np.uint8)]
