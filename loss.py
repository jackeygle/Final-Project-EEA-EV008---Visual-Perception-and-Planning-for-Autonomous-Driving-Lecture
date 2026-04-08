import torch
import torch.nn.functional as F

_SMALL_INSTANCE_THRESHOLD = 4096  # pixels — same order as official DeepLab2


def _topk_cross_entropy(logits, targets, ignore_index=255, top_k_percent=1.0,
                        pixel_weights=None):
    """Cross-entropy with optional OHEM (top-k hardest pixels) and per-pixel weights."""
    B, C, H, W = logits.shape
    per_pixel = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction="none")

    if pixel_weights is not None:
        per_pixel = per_pixel * pixel_weights

    valid = targets != ignore_index
    if not valid.any():
        return per_pixel.sum() * 0.0

    if top_k_percent < 1.0:
        per_sample_losses = []
        for b in range(B):
            vals = per_pixel[b][valid[b]]
            if vals.numel() == 0:
                per_sample_losses.append(vals.sum())
                continue
            k = max(1, int(round(vals.numel() * top_k_percent)))
            topk_vals, _ = torch.topk(vals, k, sorted=False)
            per_sample_losses.append(topk_vals.mean())
        return torch.stack(per_sample_losses).mean()

    return per_pixel[valid].mean()


def compute_loss(
    predictions,
    targets,
    offset_weights,
    motion_weights,
    semantic_weights=None,
    aux_semantic_weight: float = 0.4,
    center_loss_weight: float = 200.0,
    offset_loss_weight: float = 0.01,
    motion_loss_weight: float = 0.01,
    top_k_percent: float = 0.2,
):
    sem_loss = _topk_cross_entropy(
        predictions["semantic_logits"],
        targets["semantic_masks"],
        ignore_index=255,
        top_k_percent=top_k_percent,
        pixel_weights=semantic_weights,
    )

    thing_mask = offset_weights > 0
    center_sq = (predictions["center_heatmap"] - targets["center_heatmaps"]) ** 2
    if thing_mask.any():
        center_loss = (center_sq * thing_mask.float()).sum() / thing_mask.float().sum()
    else:
        center_loss = center_sq.mean()

    inst_reg_diff = torch.abs(predictions["center_offsets"] - targets["center_offsets"])
    inst_reg_loss = torch.mean(inst_reg_diff * offset_weights)

    motion_reg_diff = torch.abs(predictions["motion_offsets"] - targets["motion_offsets"])
    motion_reg_loss = torch.mean(motion_reg_diff * motion_weights)

    total_loss = (
        sem_loss
        + (center_loss_weight * center_loss)
        + (offset_loss_weight * inst_reg_loss)
        + (motion_loss_weight * motion_reg_loss)
    )

    sem_aux_loss = predictions["semantic_logits"].new_tensor(0.0)
    if aux_semantic_weight > 0 and "semantic_logits_aux" in predictions:
        aux = predictions["semantic_logits_aux"]
        h, w = aux.shape[2], aux.shape[3]
        sem_ds = (
            F.interpolate(
                targets["semantic_masks"].unsqueeze(1).float(),
                size=(h, w),
                mode="nearest",
            )
            .squeeze(1)
            .long()
        )
        sw_ds = None
        if semantic_weights is not None:
            sw_ds = F.interpolate(
                semantic_weights.unsqueeze(1).float(), size=(h, w), mode="nearest"
            ).squeeze(1)
        sem_aux_loss = _topk_cross_entropy(
            aux, sem_ds, ignore_index=255, top_k_percent=top_k_percent,
            pixel_weights=sw_ds,
        )
        total_loss = total_loss + aux_semantic_weight * sem_aux_loss

    return total_loss, sem_loss, center_loss, inst_reg_loss, motion_reg_loss, sem_aux_loss


def compute_semantic_pretrain_loss(
    predictions, semantic_masks, aux_semantic_weight: float = 0.4,
    top_k_percent: float = 0.2,
):
    """Encoder + semantic (+ optional res4 aux) only; for Cityscapes pretrain."""
    sem_loss = _topk_cross_entropy(
        predictions["semantic_logits"],
        semantic_masks,
        ignore_index=255,
        top_k_percent=top_k_percent,
    )
    total_loss = sem_loss
    sem_aux_loss = predictions["semantic_logits"].new_tensor(0.0)
    if aux_semantic_weight > 0 and "semantic_logits_aux" in predictions:
        aux = predictions["semantic_logits_aux"]
        h, w = aux.shape[2], aux.shape[3]
        sem_ds = (
            F.interpolate(
                semantic_masks.unsqueeze(1).float(),
                size=(h, w),
                mode="nearest",
            )
            .squeeze(1)
            .long()
        )
        sem_aux_loss = _topk_cross_entropy(
            aux, sem_ds, ignore_index=255, top_k_percent=top_k_percent,
        )
        total_loss = total_loss + aux_semantic_weight * sem_aux_loss
    return total_loss, sem_loss, sem_aux_loss


def generate_motion_targets(current_inst, prev_inst, sigma=8.0):
    """Compute motion offsets: for each pixel in current frame, offset to the
    center of the *same* instance in the previous frame.

    Returns (motion_offsets [B,2,H,W], motion_weights [B,1,H,W],
             prev_heatmap [B,1,H,W]).
    """
    B, H, W = current_inst.shape
    device = current_inst.device

    motion_offsets = torch.zeros((B, 2, H, W), device=device, dtype=torch.float32)
    motion_weights = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    prev_heatmap = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)

    y_coord, x_coord = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )

    for b in range(B):
        prev_ids = torch.unique(prev_inst[b])
        for uid in prev_ids:
            if uid == 0 or uid == 255:
                continue
            prev_mask = prev_inst[b] == uid
            prev_cy = y_coord[prev_mask].mean()
            prev_cx = x_coord[prev_mask].mean()

            dist_sq = (y_coord - prev_cy) ** 2 + (x_coord - prev_cx) ** 2
            prev_heatmap[b, 0] = torch.maximum(
                prev_heatmap[b, 0], torch.exp(-dist_sq / (2 * sigma ** 2))
            )

            curr_mask = current_inst[b] == uid
            if not curr_mask.any():
                continue
            cur_ys = y_coord[curr_mask]
            cur_xs = x_coord[curr_mask]
            motion_offsets[b, 0, curr_mask] = prev_cy - cur_ys
            motion_offsets[b, 1, curr_mask] = prev_cx - cur_xs
            motion_weights[b, 0, curr_mask] = 1.0

    return motion_offsets, motion_weights, prev_heatmap


def generate_panoptic_targets(
    instance_masks,
    semantic_masks=None,
    sigma=8.0,
    small_instance_weight=3.0,
):
    """Converts raw instance ID masks into center heatmaps, regression offsets,
    and per-pixel semantic weights (upweighting small instances)."""
    B, H, W = instance_masks.shape
    device = instance_masks.device

    center_heatmaps = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    center_offsets = torch.zeros((B, 2, H, W), device=device, dtype=torch.float32)
    offset_weights = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    semantic_weights = torch.ones((B, H, W), device=device, dtype=torch.float32)

    y_coord, x_coord = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )

    for b in range(B):
        unique_ids = torch.unique(instance_masks[b])

        for uid in unique_ids:
            if uid == 0 or uid == 255:
                continue
            mask = (instance_masks[b] == uid)
            area = int(mask.sum().item())

            y_pixels = y_coord[mask]
            x_pixels = x_coord[mask]
            center_y = y_pixels.mean()
            center_x = x_pixels.mean()

            dist_sq = (y_coord - center_y)**2 + (x_coord - center_x)**2
            gaussian = torch.exp(-dist_sq / (2 * sigma**2))
            center_heatmaps[b, 0] = torch.maximum(center_heatmaps[b, 0], gaussian)

            center_offsets[b, 0, mask] = center_y - y_pixels
            center_offsets[b, 1, mask] = center_x - x_pixels

            offset_weights[b, 0, mask] = 1.0

            if area < _SMALL_INSTANCE_THRESHOLD:
                semantic_weights[b][mask] = small_instance_weight

    return center_heatmaps, center_offsets, offset_weights, semantic_weights
