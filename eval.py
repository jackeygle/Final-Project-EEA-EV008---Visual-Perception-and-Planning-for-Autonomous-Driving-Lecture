"""Generate evaluation visualizations: a result PNG and an MP4 video.

Produces a 2×2 grid per frame:
  - Input Frame | Semantic Overlay (Cityscapes palette)
  - Instance Center Heatmap | Motion Vectors (HSV)

Usage:
  python eval.py --ckpt motion_deeplab_epoch_200.pth --sequence 0002 --num_frames 200
"""

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.amp import autocast

from dataset import KittiStepDataset
from model import MotionDeepLab
from official_postprocess import decode_panoptic_official

THING_CLASSES_KITTI_STEP = [11, 13]

CITYSCAPES_COLORS = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0],
]
_CMAP = ListedColormap(np.array(CITYSCAPES_COLORS, dtype=np.float32) / 255.0)


def _visualize(image_chw, predictions):
    sem_pred = np.argmax(predictions["semantic_logits"][0].cpu().numpy(), axis=0)
    sem_pred[sem_pred == 255] = 19

    center_heat = torch.sigmoid(predictions["center_heatmap"][0, 0]).cpu().numpy()

    motion_yx = predictions["motion_offsets"][0].cpu().numpy()
    mag, ang = cv2.cartToPolar(motion_yx[1], motion_yx[0])
    hsv = np.zeros((*motion_yx.shape[1:], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    motion_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    img_np = image_chw.permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Input Frame")
    axes[0, 1].imshow(img_np)
    axes[0, 1].imshow(sem_pred, cmap=_CMAP, alpha=0.5, vmin=0, vmax=19)
    axes[0, 1].set_title("Semantic Overlay")
    axes[1, 0].imshow(center_heat, cmap="magma")
    axes[1, 0].set_title("Instance Center Heatmap")
    axes[1, 1].imshow(motion_rgb)
    axes[1, 1].set_title("Motion Vectors (HSV)")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    return fig


def _fig_to_bgr(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


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
            fig = _visualize(curr_rgb, predictions)

            # Save a single representative frame as PNG (frame at ~30% of sequence)
            if not saved_result_png and (i - start_idx) >= (end_idx - start_idx) * 0.3:
                fig.savefig(result_path, dpi=150, bbox_inches="tight")
                print(f"Saved {result_path}")
                saved_result_png = True

            frame_bgr = _fig_to_bgr(fig)
            if video_writer is None:
                h, w, _ = frame_bgr.shape
                video_writer = cv2.VideoWriter(
                    video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h),
                )
            video_writer.write(frame_bgr)
            plt.close(fig)

            if (i - start_idx + 1) % 50 == 0 or i == end_idx - 1:
                print(f"  Processed {i - start_idx + 1}/{end_idx - start_idx} frames")

    if video_writer:
        video_writer.release()
    print(f"Video saved: {video_path}")


if __name__ == "__main__":
    main()