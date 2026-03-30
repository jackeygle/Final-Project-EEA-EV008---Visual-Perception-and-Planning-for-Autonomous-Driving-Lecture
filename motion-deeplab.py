import os, random, warnings
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import  torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import functional as TF

SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    print("cuda is available")
    torch.cuda.manual_seed_all(SEED)

print('PyTorch    :', torch.__version__)
print('torchvision:', torchvision.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device     :', device)

class _Tools:
    def save_model(self, model, path, confirm=False):
        torch.save(model.state_dict(), path); print(f'Saved {path}')
    def load_model(self, model, path, device):
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device); print(f'Loaded {path}')
tools = _Tools()

class KittiStepDataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=(385, 1249)):
        """
        Args:
            root_dir: Path to KITTI_STEP_ROOT
            split: 'train' or 'val'
            image_size: (heigh, width) to resize inputs to.
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        self.img_dir = os.path.join(root_dir, 'images', split)
        self.panoptic_dir = os.path.join(root_dir, 'panoptic_maps', split)

        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.samples = []

        for sequence_id in sorted(os.listdir(self.img_dir)):
            seq_path = os.path.join(self.img_dir, sequence_id)
            if not os.path.isdir(seq_path): continue

            frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
            for frame_name in frames:
                self.samples.append({
                    'sequence_id': sequence_id,
                    'frame_name': frame_name
                })
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq_id = sample['sequence_id']
        frame_name = sample['frame_name']

        frame_idx = int(frame_name.replace('.png', ''))
        curr_img_path = os.path.join(self.img_dir, seq_id, frame_name)

        prev_frame_name = f"{(frame_idx - 1):06d}.png"
        prev_img_path = os.path.join(self.img_dir, seq_id, prev_frame_name)

        if not os.path.exists(prev_img_path):
            prev_img_path = curr_img_path

        curr_img = Image.open(curr_img_path).convert('RGB')
        prev_img = Image.open(prev_img_path).convert('RGB')
        
        fixed_size = (384, 1248)

        # Resize inputs
        curr_img = TF.resize(curr_img, fixed_size)
        prev_img = TF.resize(prev_img, fixed_size)

        panoptic_path = os.path.join(self.panoptic_dir, seq_id, frame_name)
        panoptic_map = Image.open(panoptic_path)
        panoptic_map = TF.resize(panoptic_map, fixed_size, interpolation=TF.InterpolationMode.NEAREST)

        curr_tensor = TF.to_tensor(curr_img)
        prev_tensor = TF.to_tensor(prev_img)

        curr_tensor = self.normalize(curr_tensor)
        prev_tensor = self.normalize(prev_tensor)

        stacked_images = torch.cat([curr_tensor, prev_tensor], dim=0)

        panoptic_np = np.array(panoptic_map, dtype=np.int32)
        semantic_map = panoptic_np[:, :, 0]
        instance_map = panoptic_np[:, :, 1] * 256 + panoptic_np[:, :, 2]

        semantic_tensor = torch.from_numpy(semantic_map).long()
        instance_tensor = torch.from_numpy(instance_map).long()

        return stacked_images, semantic_tensor, instance_tensor
    

KITTI_STEP_ROOT = '.'
train_ds = KittiStepDataset(root_dir=KITTI_STEP_ROOT, split='train')

BATCH_SIZE = 4
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)


for stacked_images, semantic_masks, instance_masks in train_loader:
    print("Batch loaded succesfully!")
    print(f"Images shape:    {stacked_images.shape}")  
    print(f"Semantics shape: {semantic_masks.shape}")  
    print(f"Instances shape: {instance_masks.shape}")  
    
    break

class MotionDeepLabResNet50(nn.Module):
    """Pretrained ResNet50 backbone modified for Motion-DeepLab."""
    def __init__(self):
        super().__init__()

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )

        with torch.no_grad():
            self.conv1.weight.copy_(
                torch.cat([original_conv1.weight, original_conv1.weight], dim=1) / 2.0
            )
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 'res2' in DeepLab
        self.layer2 = resnet.layer2 # 'res3'
        self.layer3 = resnet.layer3 # 'res4'
        self.layer4 = resnet.layer4 # 'res5'
    
    def forward(self, x):
        # x expected shape: (Batch, 6, Height, Width)

        # Stem
        x = self.conv1(x)
        x = self.maxpool(self.relu(self.bn1(x)))

        res2 = self.layer1(x)
        res3 = self.layer2(res2)
        res4 = self.layer3(res3)
        res5 = self.layer4(res4)

        return {
            'res2': res2,
            'res3': res3,
            'res5': res5
        }
