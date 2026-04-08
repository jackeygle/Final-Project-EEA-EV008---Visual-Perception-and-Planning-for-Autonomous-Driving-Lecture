# dataset.py
import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

# Official DeepLab2 scale augmentation parameters
_SCALE_FACTORS = [round(s, 1) for s in np.arange(0.5, 2.05, 0.1).tolist()]


def _random_scale_and_crop(images, panoptic_maps, crop_h, crop_w):
    """Apply random scale (0.5x–2.0x) then random crop, matching official augmentation."""
    scale = random.choice(_SCALE_FACTORS)

    orig_h, orig_w = images[0].size[1], images[0].size[0]  # PIL is (W, H)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))

    scaled_imgs = [TF.resize(im, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
                   for im in images]
    scaled_pans = [TF.resize(pm, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)
                   for pm in panoptic_maps]

    # Pad if smaller than crop size (ignore_label=255 for semantic channel)
    pad_h = max(crop_h - new_h, 0)
    pad_w = max(crop_w - new_w, 0)
    if pad_h > 0 or pad_w > 0:
        scaled_imgs = [TF.pad(im, (0, 0, pad_w, pad_h), fill=0) for im in scaled_imgs]
        # Pad panoptic with (255, 0, 0) so semantic=255 (ignore) and instance=0
        scaled_pans = [TF.pad(pm, (0, 0, pad_w, pad_h), fill=255) for pm in scaled_pans]

    padded_h = max(new_h, crop_h)
    padded_w = max(new_w, crop_w)

    top = random.randint(0, padded_h - crop_h)
    left = random.randint(0, padded_w - crop_w)

    cropped_imgs = [TF.crop(im, top, left, crop_h, crop_w) for im in scaled_imgs]
    cropped_pans = [TF.crop(pm, top, left, crop_h, crop_w) for pm in scaled_pans]

    return cropped_imgs, cropped_pans


class KittiStepDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        image_size=(385, 1249),
        multi_scale=True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.multi_scale = multi_scale and (split == "train")
        self.img_dir = os.path.join(root_dir, 'images', split)
        self.panoptic_dir = os.path.join(root_dir, 'panoptic_maps', split)

        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.samples = []

        for sequence_id in sorted(os.listdir(self.img_dir)):
            seq_path = os.path.join(self.img_dir, sequence_id)
            if not os.path.isdir(seq_path):
                continue
            frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
            for frame_name in frames:
                self.samples.append({
                    'sequence_id': sequence_id,
                    'frame_name': frame_name,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq_id = sample['sequence_id']
        frame_name = sample['frame_name']

        frame_idx = int(frame_name.replace('.png', ''))
        curr_img_path = os.path.join(self.img_dir, seq_id, frame_name)
        curr_panoptic_path = os.path.join(self.panoptic_dir, seq_id, frame_name)

        prev_frame_name = f"{(frame_idx - 1):06d}.png"
        prev_img_path = os.path.join(self.img_dir, seq_id, prev_frame_name)
        prev_panoptic_path = os.path.join(self.panoptic_dir, seq_id, prev_frame_name)

        if not os.path.exists(prev_img_path):
            prev_img_path = curr_img_path
            prev_panoptic_path = curr_panoptic_path

        curr_img = Image.open(curr_img_path).convert('RGB')
        prev_img = Image.open(prev_img_path).convert('RGB')
        curr_panoptic_map = Image.open(curr_panoptic_path)
        prev_panoptic_map = Image.open(prev_panoptic_path)

        crop_h, crop_w = self.image_size

        if self.multi_scale:
            # Official-style: random scale (0.5x-2.0x) → random crop → random flip
            [curr_img, prev_img], [curr_panoptic_map, prev_panoptic_map] = \
                _random_scale_and_crop(
                    [curr_img, prev_img],
                    [curr_panoptic_map, prev_panoptic_map],
                    crop_h, crop_w,
                )
        else:
            fixed_size = (crop_h, crop_w)
            curr_img = TF.resize(curr_img, fixed_size, interpolation=TF.InterpolationMode.BILINEAR)
            prev_img = TF.resize(prev_img, fixed_size, interpolation=TF.InterpolationMode.BILINEAR)
            curr_panoptic_map = TF.resize(curr_panoptic_map, fixed_size,
                                          interpolation=TF.InterpolationMode.NEAREST)
            prev_panoptic_map = TF.resize(prev_panoptic_map, fixed_size,
                                          interpolation=TF.InterpolationMode.NEAREST)

        if self.split == "train" and random.random() < 0.5:
            curr_img = TF.hflip(curr_img)
            prev_img = TF.hflip(prev_img)
            curr_panoptic_map = TF.hflip(curr_panoptic_map)
            prev_panoptic_map = TF.hflip(prev_panoptic_map)

        curr_tensor = self.normalize(TF.to_tensor(curr_img))
        prev_tensor = self.normalize(TF.to_tensor(prev_img))
        stacked_images = torch.cat([curr_tensor, prev_tensor], dim=0)

        panoptic_np = np.array(curr_panoptic_map, dtype=np.int32)
        semantic_map = panoptic_np[:, :, 0]
        instance_map = panoptic_np[:, :, 1] * 256 + panoptic_np[:, :, 2]

        prev_panoptic_np = np.array(prev_panoptic_map, dtype=np.int32)
        prev_instance_map = prev_panoptic_np[:, :, 1] * 256 + prev_panoptic_np[:, :, 2]

        semantic_tensor = torch.from_numpy(semantic_map).long()
        instance_tensor = torch.from_numpy(instance_map).long()
        prev_instance_tensor = torch.from_numpy(prev_instance_map).long()

        return stacked_images, semantic_tensor, instance_tensor, prev_instance_tensor

