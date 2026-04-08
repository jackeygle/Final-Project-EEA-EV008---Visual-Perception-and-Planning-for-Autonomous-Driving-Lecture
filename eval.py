"""Generate evaluation visualizations: a result PNG and an MP4 video.

Produces a 2x2 grid per frame (pure OpenCV, no matplotlib):
  - Input Frame | Semantic Overlay (Cityscapes palette)
  - Instance Center Heatmap | Motion Vectors (HSV)

Usage:
  python eval.py --ckpt motion_deeplab_epoch_200.pth --sequence 0002 --num_frames 200
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torch.amp import autocast

from dataset import KittiStepDataset
from model import MotionDeepLab
from official_postprocess import decode_panoptic_official

THING_CLASSES_KITTI_STEP = [11, 13]

CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0],
], dtype=np.uint8)

MAGMA_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_MAGMA)


def _render_frame_cv2(img_rgb_uint8, predictions, h, w):
    """Render 2x2 grid as a single BGR image using OpenCV."""
    sem_pred = torch.argmax(predictions["semantic_logits"][0], dim=0).cpu().numpy()
    sem_pred = np.clip(sem_pred, 0, 19).astype(np.uint8)

    sem_color = CITYSCAPES_PALETTE[sem_pred]
    overlay = cv2.addWeighted(img_rgb_uint8, 0.5, sem_color, 0.5, 0)

    center_heat = torch.sigmoid(predictions["center_heatmap"][0, 0]).cpu().numpy()
    heat_u8 = np.clip(center_heat * 255, 0, 255).astype(np.uint8)
    heat_color = MAGMA_LUT[heat_u8].reshape(h, w, 3)

    motion_yx = predictions["motion_offsets"][0].cpu().numpy()
    mag, ang = cv2.cartToPolar(motion_yx[1], motion_yx[0])
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    motion_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    label_h = 30
    def _add_label(img, text):
        out = np.zeros((img.shape[0] + label_h, img.shape[1], 3), dtype=np.uint8)
        out[:label_h] = 40
        cv2.putText(out, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out[label_h:] = img
        return out

    tl = _add_label(img_bgr, "Input Frame")
    tr = _add_label(overlay_bgr, "Semantic Overlay")
    bl = _add_label(heat_color, "Instance Center Heatmap")
    br = _add_label(motion_bgr, "Motion Vectors (HSV)")

    top = np.concatenate([tl, tr], axis=1)
    bot = np.concatenate([bl, br], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    return grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=".")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--sequence", type=str, default="0002")
    p.add_argument("--num_frames", type=int, default=200)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--crop_h", type=int, default=385)
    p.add_argument("--crop_w", type=int, default=1249)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionDeepLab().to(device)
    if not os.path.exists(args.ckpt):
        print(f"Error: checkpoint not found at '{args.ckpt}'")
        sys.exit(1)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    print(f"Loaded {args.ckpt}")
    model.eval()

    val_ds = KittiStepDataset(
        root_dir=args.data_root, split="val",
        image_size=(args.crop_h, args.crop_w), multi_scale=False,
    )

    start_idx = None
    for idx, sample in enumerate(val_ds.samples):
        if sample["sequence_id"] == args.sequence:
            start_idx = idx
            break
    if start_idx is None:
        print(f"Error: sequence {args.sequence} not found in val set!")
        sys.exit(1)

    end_idx = min(start_idx + args.num_frames, len(val_ds))
    print(f"Sequence {args.sequence}: frames {start_idx}..{end_idx - 1} ({end_idx - start_idx} frames)")

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    prev_heatmap = torch.zeros((1, 1, args.crop_h, args.crop_w), device=device)
    video_writer = None
    saved_result_png = False

    video_path = os.path.join(args.output_dir, f"evaluation_video_{args.sequence}.mp4")
    result_path = os.path.join(args.output_dir, f"evaluation_result_{args.sequence}.png")

    with torch.no_grad():
        for i in range(start_idx, end_idx):
            stacked_images, _, _, _ = val_ds[i]
            images = stacked_images.unsqueeze(0).to(device)
            model_input = torch.cat([images, prev_heatmap], dim=1)

            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                predictions = model(model_input)

            _, rendered_hw, _ = decode_panoptic_official(
                predictions["semantic_logits"][0],
                predictions["center_heatmap"][0],
                predictions["center_offsets"][0],
                thing_class_ids=THING_CLASSES_KITTI_STEP,
                label_divisor=1000,
                void_label=255,
                center_threshold=0.1,
                nms_kernel=13,
                keep_k_centers=200,
                stuff_area_limit=0,
            )
            prev_heatmap = torch.from_numpy(rendered_hw).unsqueeze(0).unsqueeze(0).to(device)

            curr_rgb = torch.clamp(images[0, :3] * std + mean, 0, 1)
            img_rgb_uint8 = (curr_rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            grid_bgr = _render_frame_cv2(img_rgb_uint8, predictions, args.crop_h, args.crop_w)

            if not saved_result_png and (i - start_idx) >= (end_idx - start_idx) * 0.3:
                cv2.imwrite(result_path, grid_bgr)
                print(f"Saved {result_path}")
                saved_result_png = True

            if video_writer is None:
                gh, gw = grid_bgr.shape[:2]
                video_writer = cv2.VideoWriter(
                    video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (gw, gh),
                )
            video_writer.write(grid_bgr)

            if (i - start_idx + 1) % 50 == 0 or i == end_idx - 1:
                print(f"  Processed {i - start_idx + 1}/{end_idx - start_idx} frames")

    if video_writer:
        video_writer.release()
    print(f"Video saved: {video_path}")


if __name__ == "__main__":
    main()
