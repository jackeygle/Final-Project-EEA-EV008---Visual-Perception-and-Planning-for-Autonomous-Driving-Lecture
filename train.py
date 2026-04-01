import  torch
import  os
from    torch.amp import autocast, GradScaler
from    model import MotionDeepLab
from    loss  import compute_loss, generate_panoptic_targets
from    dataset import KittiStepDataset
from    torch.utils.data import DataLoader

KITTI_STEP_ROOT = '.'
train_ds = KittiStepDataset(root_dir=KITTI_STEP_ROOT, split='train')

BATCH_SIZE = 4
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

for stacked_images, semantic_masks, instance_masks, prev_inst_masks in train_loader:
    print("Batch loaded succesfully!")
    print(f"Images shape:    {stacked_images.shape}")  
    print(f"Semantics shape: {semantic_masks.shape}")  
    print(f"Instances shape: {instance_masks.shape}") 
    print(f"Prev Instances shape: {prev_inst_masks.shape}")  

    break

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MotionDeepLab().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = GradScaler()

EPOCHS = 10
ACCUMULATION_STEPS = 4
CURRENT_MODEL = 'motion_deeplab_epoch_11.pth'

RESUME_TRAINING = True

if RESUME_TRAINING and os.path.exists(CURRENT_MODEL):
    checkpoint = torch.load(CURRENT_MODEL)
    model.load_state_dict(checkpoint)
    start_epoch = 11
    print(f"Resuming from {CURRENT_MODEL} at Epoch {start_epoch}")
else:
    start_epoch = 1
    print("Starting training from scratch...")

print("Starting training!")
for epoch in range(start_epoch, start_epoch + EPOCHS):
    model.train()
    optimizer.zero_grad()

    for i, (images, sem_masks, inst_masks, prev_inst_masks) in enumerate(train_loader):
        images = images.to(device)
        sem_masks = sem_masks.to(device)
        inst_masks = inst_masks.to(device)
        prev_inst_masks = prev_inst_masks.to(device)

        gt_heatmaps, gt_inst_offsets, offset_weights = generate_panoptic_targets(inst_masks)
        
        # We need the motion targets as well (distance from current pixel to Previous frame)
        prev_heatmaps, gt_motion_offsets, _ = generate_panoptic_targets(prev_inst_masks)

        targets = {
            'semantic_masks': sem_masks,
            'center_heatmaps': gt_heatmaps,
            'center_offsets': gt_inst_offsets,
            'motion_offsets': gt_motion_offsets
        }
        model_input = torch.cat([images, prev_heatmaps], dim=1)

        with autocast(device_type='cuda'):
            predictions = model(model_input)
            total_loss, _, _, _, _ = compute_loss(predictions, targets, offset_weights)

            total_loss = total_loss / ACCUMULATION_STEPS

        scaler.scale(total_loss).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    print(f"Epoch {epoch} Complete.")
    torch.save(model.state_dict(), f'motion_deeplab_epoch_{epoch}.pth')
    print(f"Model saved to motion_deeplab_epoch_{epoch}.pth")
