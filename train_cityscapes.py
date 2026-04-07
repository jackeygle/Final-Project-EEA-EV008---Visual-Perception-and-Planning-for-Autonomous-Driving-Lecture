import argparse
import os

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset_cityscapes import CityscapesSemSegDataset
from loss import compute_semantic_pretrain_loss
from model import MotionDeepLab


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
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CityscapesSemSegDataset(
        root_dir=args.data_root,
        split=args.split,
        image_size=(args.image_h, args.image_w),
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

    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0

        for i, (images, sem_masks) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            sem_masks = sem_masks.to(device, non_blocking=True)

            prev_heat = torch.zeros(
                (images.shape[0], 1, images.shape[2], images.shape[3]),
                dtype=images.dtype,
                device=device,
            )
            model_input = torch.cat([images, prev_heat], dim=1)

            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                preds = model(model_input)
                total, sem, aux = compute_semantic_pretrain_loss(
                    preds,
                    sem_masks,
                    aux_semantic_weight=args.aux_semantic_weight,
                )
                loss = total / args.accumulation_steps

            scaler.scale(loss).backward()
            if (i + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += float(total.detach())

        print(
            f"Epoch {epoch} | loss {running / max(1, len(loader)):.4f} "
            f"(aux_weight={args.aux_semantic_weight})"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = os.path.join(args.save_dir, f"motion_deeplab_cityscapes_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"Saved {ckpt}")


if __name__ == "__main__":
    main()
