import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random
from model import DualBackboneTimeLapseClassifier
import torch

class DualTimeLapseDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train', 
        context_size: int = 5,
        reference_size: int = 3,
        transform=None
    ):
        """
        Dataset for dual-backbone time-lapse classification
        
        Args:
            root_dir: Root directory containing train/test folders
            split: 'train' or 'test'
            context_size: Number of sequential context frames
            reference_size: Number of reference frames for comparison
            transform: Image transformations
        """
        self.root_dir = root_dir
        self.split = split
        self.context_size = context_size
        self.reference_size = reference_size
        self.transform = transform or self._default_transforms()
        
        info_path = os.path.join(root_dir, split, 'info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.info = json.load(f)
        else:
            self.info = {}
        
        self.samples = self._build_samples()
        
        self.class_to_idx = {'normal': 0, 'abnormal': 1}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    
    def _default_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _build_samples(self):
        samples = []
        split_dir = os.path.join(self.root_dir, self.split)
        
        for class_name in ['normal', 'abnormal']:
            class_dir = os.path.join(split_dir, class_name, 'images')
            if not os.path.exists(class_dir):
                continue
            
            # Get all image files sorted by name (assuming temporal order)
            image_files = sorted([f for f in os.listdir(class_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # Create sequences with both context and reference frames
            min_required = self.context_size + self.reference_size
            if len(image_files) < min_required:
                continue
                
            for i in range(len(image_files) - min_required + 1):
                # Context frames: sequential frames
                context_frames = []
                for j in range(self.context_size):
                    img_path = os.path.join(class_dir, image_files[i + j])
                    context_frames.append(img_path)
                
                # Reference frames: early frames from the sequence or from normal class
                if class_name == 'abnormal' and self.split == 'train':
                    # For abnormal samples, use normal frames as reference
                    normal_dir = os.path.join(split_dir, 'normal', 'images')
                    if os.path.exists(normal_dir):
                        normal_files = sorted([f for f in os.listdir(normal_dir)
                                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        if len(normal_files) >= self.reference_size:
                            reference_frames = []
                            ref_start = random.randint(0, len(normal_files) - self.reference_size)
                            for j in range(self.reference_size):
                                ref_path = os.path.join(normal_dir, normal_files[ref_start + j])
                                reference_frames.append(ref_path)
                        else:
                            # Fallback: use early frames from the same sequence
                            reference_frames = context_frames[:self.reference_size]
                    else:
                        reference_frames = context_frames[:self.reference_size]
                else:
                    # For normal samples or test set, use early frames from same sequence
                    reference_frames = []
                    for j in range(self.reference_size):
                        ref_idx = min(j, len(image_files) - 1)
                        img_path = os.path.join(class_dir, image_files[ref_idx])
                        reference_frames.append(img_path)
                
                samples.append({
                    'context_frames': context_frames,
                    'reference_frames': reference_frames,
                    'label': class_name,
                    'class_idx': self.class_to_idx[class_name]
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        context_sequence = []
        for img_path in sample['context_frames']:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            context_sequence.append(image)
        
        reference_sequence = []
        for img_path in sample['reference_frames']:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            reference_sequence.append(image)
        
        context_tensor = torch.stack(context_sequence, dim=0)  # (context_size, 3, H, W)
        reference_tensor = torch.stack(reference_sequence, dim=0)  # (reference_size, 3, H, W)
        
        return context_tensor, reference_tensor, sample['class_idx']

def create_dual_model_and_dataloader(
    data_root: str, 
    context_size: int = 5, 
    reference_size: int = 3,
    batch_size: int = 16
):
    model = DualBackboneTimeLapseClassifier(
        num_classes=2,
        context_size=context_size,
        reference_size=reference_size,
        context_encoder="efficientnet-b0",
        reference_encoder="efficientnet-b0",
        context_encoding_size=512,
        reference_encoding_size=512,
        mha_num_attention_heads=8,
        mha_num_attention_layers=4,
        dropout_rate=0.1
    )
    
    train_dataset = DualTimeLapseDataset(
        data_root, split='train', 
        context_size=context_size, 
        reference_size=reference_size
    )
    test_dataset = DualTimeLapseDataset(
        data_root, split='test', 
        context_size=context_size, 
        reference_size=reference_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return model, train_loader, test_loader
