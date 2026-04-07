import argparse
import os

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset import KittiStepDataset
from loss import compute_loss, generate_motion_targets, generate_panoptic_targets
from model import MotionDeepLab


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=".")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume_ckpt", type=str, default="")
    p.add_argument("--start_epoch", type=int, default=1)
    p.add_argument("--save_dir", type=str, default=".")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop_h", type=int, default=385)
    p.add_argument("--crop_w", type=int, default=1249)
    p.add_argument("--aux_semantic_weight", type=float, default=0.4)
    p.add_argument("--center_loss_weight", type=float, default=200.0)
    p.add_argument("--offset_loss_weight", type=float, default=0.01)
    p.add_argument("--motion_loss_weight", type=float, default=0.01)
    args = p.parse_args()

    train_ds = KittiStepDataset(
        root_dir=args.data_root,
        split="train",
        image_size=(args.crop_h, args.crop_w),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionDeepLab().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    start_epoch = args.start_epoch
    if args.resume and args.resume_ckpt and os.path.exists(args.resume_ckpt):
        checkpoint = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Resuming from {args.resume_ckpt} at epoch {start_epoch}")
    else:
        print("Starting training from scratch...")

    os.makedirs(args.save_dir, exist_ok=True)
    print("Starting training")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for i, (images, sem_masks, inst_masks, prev_inst_masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            sem_masks = sem_masks.to(device, non_blocking=True)
            inst_masks = inst_masks.to(device, non_blocking=True)
            prev_inst_masks = prev_inst_masks.to(device, non_blocking=True)

            gt_heatmaps, gt_inst_offsets, offset_weights = generate_panoptic_targets(
                inst_masks
            )
            gt_motion_offsets, motion_weights, prev_heatmaps = generate_motion_targets(
                inst_masks, prev_inst_masks
            )

            targets = {
                "semantic_masks": sem_masks,
                "center_heatmaps": gt_heatmaps,
                "center_offsets": gt_inst_offsets,
                "motion_offsets": gt_motion_offsets,
            }
            model_input = torch.cat([images, prev_heatmaps], dim=1)

            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                predictions = model(model_input)
                total_loss, _, _, _, _, _ = compute_loss(
                    predictions,
                    targets,
                    offset_weights,
                    motion_weights,
                    aux_semantic_weight=args.aux_semantic_weight,
                    center_loss_weight=args.center_loss_weight,
                    offset_loss_weight=args.offset_loss_weight,
                    motion_loss_weight=args.motion_loss_weight,
                )
                total_loss = total_loss / args.accumulation_steps

            scaler.scale(total_loss).backward()

            if (i + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        print(f"Epoch {epoch} complete.")
        if epoch % args.save_every == 0 or epoch == start_epoch + args.epochs - 1:
            out = os.path.join(args.save_dir, f"motion_deeplab_epoch_{epoch}.pth")
            torch.save(model.state_dict(), out)
            print(f"Saved {out}")


if __name__ == "__main__":
    main()
