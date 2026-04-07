# dataset_cityscapes.py — Cityscapes fine semantic segmentation for encoder/decoder pretrain
import os
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from cityscapes_labels import labelids_to_trainids


class CityscapesSemSegDataset(Dataset):
    """
    Reads official Cityscapes layout:
      {root}/leftImg8bit/{split}/<city>/*_leftImg8bit.png
      {root}/gtFine/{split}/<city>/*_gtFine_labelTrainIds.png  (preferred)
      or *_gtFine_labelIds.png (converted via cityscapes_labels).

    Returns the same image tensor layout as KITTI: 6ch (curr||prev RGB), prev=curr;
    KITTI training adds a 7th channel (prev heatmap) in the training loop.
    """

    def __init__(self, root_dir: str, split: str = "train", image_size=(384, 1248)):
        self.root_dir = os.path.abspath(root_dir)
        self.split = split
        self.image_size = image_size
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
            if train_path.is_file():
                self.samples.append((str(img_path), str(train_path), "trainid"))
            elif ids_path.is_file():
                self.samples.append((str(img_path), str(ids_path), "labelid"))

        if not self.samples:
            raise RuntimeError(
                f"No paired (image, label) under {img_root} / {gt_root}. "
                "Check split name and that gtFine labels exist."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path, lbl_kind = self.samples[idx]
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
        return stacked, sem
