import argparse
import os

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset_cityscapes import CityscapesSemSegDataset
from loss import (
    compute_semantic_pretrain_loss,
    generate_panoptic_targets,
    _topk_cross_entropy,
)
from model import MotionDeepLab


def _poly_lr_lambda(step, total_steps, power=0.9):
    return max(0.0, (1.0 - step / total_steps) ** power)


def _panoptic_pretrain_loss(predictions, sem_masks, inst_masks,
                            aux_semantic_weight=0.4, center_loss_weight=200.0,
                            offset_loss_weight=0.01, top_k_percent=0.2,
                            small_instance_weight=3.0):
    """Combined semantic + instance (center + offset) loss for Cityscapes."""
    gt_heatmaps, gt_offsets, offset_weights, semantic_weights = \
        generate_panoptic_targets(
            inst_masks, sem_masks, small_instance_weight=small_instance_weight,
        )

    sem_loss = _topk_cross_entropy(
        predictions["semantic_logits"], sem_masks,
        ignore_index=255, top_k_percent=top_k_percent,
        pixel_weights=semantic_weights,
    )

    thing_mask = offset_weights > 0
    center_sq = (predictions["center_heatmap"] - gt_heatmaps) ** 2
    if thing_mask.any():
        center_loss = (center_sq * thing_mask.float()).sum() / thing_mask.float().sum()
    else:
        center_loss = center_sq.mean()

    inst_reg_diff = torch.abs(predictions["center_offsets"] - gt_offsets)
    inst_reg_loss = torch.mean(inst_reg_diff * offset_weights)

    total = sem_loss + center_loss_weight * center_loss + offset_loss_weight * inst_reg_loss

    sem_aux_loss = predictions["semantic_logits"].new_tensor(0.0)
    if aux_semantic_weight > 0 and "semantic_logits_aux" in predictions:
        aux = predictions["semantic_logits_aux"]
        h, w = aux.shape[2], aux.shape[3]
        sem_ds = F.interpolate(
            sem_masks.unsqueeze(1).float(), size=(h, w), mode="nearest"
        ).squeeze(1).long()
        sw_ds = F.interpolate(
            semantic_weights.unsqueeze(1).float(), size=(h, w), mode="nearest"
        ).squeeze(1)
        sem_aux_loss = _topk_cross_entropy(
            aux, sem_ds, ignore_index=255, top_k_percent=top_k_percent,
            pixel_weights=sw_ds,
        )
        total = total + aux_semantic_weight * sem_aux_loss

    return total, sem_loss, center_loss, inst_reg_loss


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_h", type=int, default=384)
    p.add_argument("--image_w", type=int, default=1248)
    p.add_argument("--aux_semantic_weight", type=float, default=0.4)
    p.add_argument("--save_dir", type=str, default="weights")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--panoptic", action="store_true", default=True,
                   help="Full-branch pretrain (semantic+instance)")
    p.add_argument("--no_panoptic", dest="panoptic", action="store_false",
                   help="Semantic-only pretrain")
    p.add_argument("--top_k_percent", type=float, default=0.2)
    p.add_argument("--lr_schedule", type=str, default="poly",
                   choices=["poly", "cosine", "none"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CityscapesSemSegDataset(
        root_dir=args.data_root,
        split=args.split,
        image_size=(args.image_h, args.image_w),
        panoptic=args.panoptic,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model = MotionDeepLab().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    steps_per_epoch = max(1, len(loader) // args.accumulation_steps)
    total_steps = steps_per_epoch * args.epochs

    if args.lr_schedule == "poly":
        scheduler = LambdaLR(optimizer, lambda s: _poly_lr_lambda(s, total_steps))
    elif args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6)
    else:
        scheduler = None

    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    mode_str = "panoptic (sem+inst)" if args.panoptic else "semantic-only"
    print(f"Cityscapes pretrain: {mode_str}, {args.epochs} epochs, lr_schedule={args.lr_schedule}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0

        for i, batch in enumerate(loader):
            if args.panoptic and len(batch) == 3:
                images, sem_masks, inst_masks = batch
                images = images.to(device, non_blocking=True)
                sem_masks = sem_masks.to(device, non_blocking=True)
                inst_masks = inst_masks.to(device, non_blocking=True)

                prev_heat = torch.zeros(
                    (images.shape[0], 1, images.shape[2], images.shape[3]),
                    dtype=images.dtype, device=device,
                )
                model_input = torch.cat([images, prev_heat], dim=1)

                with autocast(device_type=device.type, enabled=device.type == "cuda"):
                    preds = model(model_input)
                    total, _, _, _ = _panoptic_pretrain_loss(
                        preds, sem_masks, inst_masks,
                        aux_semantic_weight=args.aux_semantic_weight,
                        top_k_percent=args.top_k_percent,
                    )
                    loss = total / args.accumulation_steps
            else:
                images, sem_masks = batch[0], batch[1]
                images = images.to(device, non_blocking=True)
                sem_masks = sem_masks.to(device, non_blocking=True)

                prev_heat = torch.zeros(
                    (images.shape[0], 1, images.shape[2], images.shape[3]),
                    dtype=images.dtype, device=device,
                )
                model_input = torch.cat([images, prev_heat], dim=1)

                with autocast(device_type=device.type, enabled=device.type == "cuda"):
                    preds = model(model_input)
                    total, _, _ = compute_semantic_pretrain_loss(
                        preds, sem_masks,
                        aux_semantic_weight=args.aux_semantic_weight,
                        top_k_percent=args.top_k_percent,
                    )
                    loss = total / args.accumulation_steps

            scaler.scale(loss).backward()
            if (i + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if scheduler is not None:
                    scheduler.step()

            running += float(total.detach())

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} | loss {running / max(1, len(loader)):.4f} "
              f"| lr {lr_now:.2e} (aux_weight={args.aux_semantic_weight})")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = os.path.join(args.save_dir, f"motion_deeplab_cityscapes_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"Saved {ckpt}")


if __name__ == "__main__":
    main()
