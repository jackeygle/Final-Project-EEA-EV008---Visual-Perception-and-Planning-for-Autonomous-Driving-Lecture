import argparse
import math
import os

import torch
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset import KittiStepDataset
from loss import compute_loss, generate_motion_targets, generate_panoptic_targets
from model import MotionDeepLab


def _poly_lr_lambda(current_step, total_steps, power=0.9):
    return max(0.0, (1.0 - current_step / total_steps) ** power)


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
    p.add_argument("--top_k_percent", type=float, default=0.2)
    p.add_argument("--small_instance_weight", type=float, default=3.0)
    p.add_argument("--lr_schedule", type=str, default="poly",
                   choices=["poly", "cosine", "none"])
    args = p.parse_args()

    train_ds = KittiStepDataset(
        root_dir=args.data_root,
        split="train",
        image_size=(args.crop_h, args.crop_w),
        multi_scale=True,
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

    steps_per_epoch = max(1, len(train_loader) // args.accumulation_steps)
    total_steps = steps_per_epoch * args.epochs

    if args.lr_schedule == "poly":
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: _poly_lr_lambda(step, total_steps),
        )
    elif args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6,
        )
    else:
        scheduler = None

    start_epoch = args.start_epoch
    if args.resume and args.resume_ckpt and os.path.exists(args.resume_ckpt):
        checkpoint = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Resuming from {args.resume_ckpt} at epoch {start_epoch}")
    else:
        print("Starting training from scratch...")

    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    print(f"Starting training: {args.epochs} epochs, {total_steps} opt steps, "
          f"lr_schedule={args.lr_schedule}, top_k={args.top_k_percent}, "
          f"small_inst_w={args.small_instance_weight}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for i, (images, sem_masks, inst_masks, prev_inst_masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            sem_masks = sem_masks.to(device, non_blocking=True)
            inst_masks = inst_masks.to(device, non_blocking=True)
            prev_inst_masks = prev_inst_masks.to(device, non_blocking=True)

            gt_heatmaps, gt_inst_offsets, offset_weights, semantic_weights = \
                generate_panoptic_targets(
                    inst_masks, sem_masks,
                    small_instance_weight=args.small_instance_weight,
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
                    semantic_weights=semantic_weights,
                    aux_semantic_weight=args.aux_semantic_weight,
                    center_loss_weight=args.center_loss_weight,
                    offset_loss_weight=args.offset_loss_weight,
                    motion_loss_weight=args.motion_loss_weight,
                    top_k_percent=args.top_k_percent,
                )
                total_loss = total_loss / args.accumulation_steps

            scaler.scale(total_loss).backward()
            running_loss += float(total_loss.detach()) * args.accumulation_steps

            if (i + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if scheduler is not None:
                    scheduler.step()

        avg_loss = running_loss / max(1, len(train_loader))
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} | loss {avg_loss:.4f} | lr {lr_now:.2e} | step {global_step}/{total_steps}")

        if epoch % args.save_every == 0 or epoch == start_epoch + args.epochs - 1:
            out = os.path.join(args.save_dir, f"motion_deeplab_epoch_{epoch}.pth")
            torch.save(model.state_dict(), out)
            print(f"Saved {out}")


if __name__ == "__main__":
    main()
