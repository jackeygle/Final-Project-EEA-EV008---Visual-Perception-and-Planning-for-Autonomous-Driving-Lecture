# dataset_cityscapes.py — Cityscapes for full-branch pretrain (semantic + instance)
import os
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from cityscapes_labels import labelids_to_trainids

# Cityscapes thing trainIds (person=11, rider=12, car=13, truck=14,
# bus=15, train=16, motorcycle=17, bicycle=18)
_CITYSCAPES_THING_TRAINIDS = {11, 12, 13, 14, 15, 16, 17, 18}


class CityscapesSemSegDataset(Dataset):
    """Cityscapes dataset supporting both semantic-only and panoptic modes.

    When panoptic=True (default), also returns instance maps derived from
    *_gtFine_instanceIds.png, enabling full-branch pretrain (semantic + center + offset).
    """

    def __init__(self, root_dir: str, split: str = "train",
                 image_size=(384, 1248), panoptic: bool = True):
        self.root_dir = os.path.abspath(root_dir)
        self.split = split
        self.image_size = image_size
        self.panoptic = panoptic
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        img_root = Path(self.root_dir) / "leftImg8bit" / split
        gt_root = Path(self.root_dir) / "gtFine" / split
        if not img_root.is_dir():
            raise FileNotFoundError(f"Missing Cityscapes images: {img_root}")
        if not gt_root.is_dir():
            raise FileNotFoundError(f"Missing Cityscapes gtFine: {gt_root}")

        self.samples = []
        for img_path in sorted(img_root.glob("*/*_leftImg8bit.png")):
            city = img_path.parent.name
            prefix = img_path.name.replace("_leftImg8bit.png", "")
            train_path = gt_root / city / f"{prefix}_gtFine_labelTrainIds.png"
            ids_path = gt_root / city / f"{prefix}_gtFine_labelIds.png"
            inst_path = gt_root / city / f"{prefix}_gtFine_instanceIds.png"
            sem_path = str(train_path) if train_path.is_file() else str(ids_path) if ids_path.is_file() else None
            if sem_path is None:
                continue
            lbl_kind = "trainid" if train_path.is_file() else "labelid"
            entry = (str(img_path), sem_path, lbl_kind,
                     str(inst_path) if inst_path.is_file() else None)
            self.samples.append(entry)

        if not self.samples:
            raise RuntimeError(
                f"No paired (image, label) under {img_root} / {gt_root}. "
                "Check split name and that gtFine labels exist."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path, lbl_kind, inst_path = self.samples[idx]
        h, w = self.image_size

        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, (h, w), interpolation=TF.InterpolationMode.BILINEAR)
        curr = self.normalize(TF.to_tensor(img))
        stacked = torch.cat([curr, curr.clone()], dim=0)

        lbl_pil = Image.open(lbl_path)
        lbl_pil = lbl_pil.resize((w, h), Image.NEAREST)
        lbl = np.array(lbl_pil, dtype=np.int32)
        if lbl_kind == "labelid":
            lbl = labelids_to_trainids(lbl).astype(np.int64)
        else:
            lbl = lbl.astype(np.int64)
            lbl[lbl == 255] = 255
        sem = torch.from_numpy(lbl).long()

        if not self.panoptic or inst_path is None:
            return stacked, sem

        # Cityscapes instanceIds encode: labelId * 1000 + instanceNum
        inst_pil = Image.open(inst_path)
        inst_pil = inst_pil.resize((w, h), Image.NEAREST)
        inst_raw = np.array(inst_pil, dtype=np.int32)
        cs_label_id = inst_raw // 1000
        cs_inst_num = inst_raw % 1000

        # Build KITTI-style instance map: unique per-pixel instance ID for things
        instance_map = np.zeros((h, w), dtype=np.int64)
        next_id = 1
        for uid in np.unique(inst_raw):
            lid = uid // 1000
            inum = uid % 1000
            if inum == 0:
                continue
            # Map Cityscapes labelId to trainId and check if thing
            tid_arr = labelids_to_trainids(np.array([lid]))
            tid = int(tid_arr[0])
            if tid not in _CITYSCAPES_THING_TRAINIDS:
                continue
            mask = inst_raw == uid
            instance_map[mask] = next_id
            next_id += 1

        inst_tensor = torch.from_numpy(instance_map).long()
        return stacked, sem, inst_tensor
