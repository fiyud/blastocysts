import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random

class EmbryoVideoDualBackboneDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, max_frames=None, 
                 context_size=5, reference_size=3, sampling_strategy='uniform'):
        """
        Dataset for dual backbone time-lapse classification
        
        Args:
            root_dir: Root directory path
            split: 'train' or 'test'
            transform: Image transformations
            max_frames: Maximum number of frames to use from each video
            context_size: Number of context frames (sequential temporal frames)
            reference_size: Number of reference frames (baseline frames)
            sampling_strategy: 'uniform', 'random', or 'temporal_split'
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.max_frames = max_frames
        self.context_size = context_size
        self.reference_size = reference_size
        self.sampling_strategy = sampling_strategy
        
        self.samples = []
        for label_name, label_id in [('normal', 0), ('abnormal', 1)]:
            split_label_dir = os.path.join(root_dir, split, label_name)
            if not os.path.exists(split_label_dir):
                continue
            for video_folder in sorted(os.listdir(split_label_dir)):
                video_folder_path = os.path.join(split_label_dir, video_folder)
                if not os.path.isdir(video_folder_path):
                    continue
                
                for sub_video_folder in sorted(os.listdir(video_folder_path)):
                    sub_img_path = os.path.join(video_folder_path, sub_video_folder)
                    if not os.path.isdir(sub_img_path):
                        continue
                    
                    self.samples.append({
                        'frames_folder': sub_img_path,
                        'label': label_id
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def _load_frames(self, frames_folder):
        frame_files = sorted([
            f for f in os.listdir(frames_folder)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        if self.max_frames:
            frame_files = frame_files[:self.max_frames]
        
        frames = []
        for f in frame_files:
            img_path = os.path.join(frames_folder, f)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        
        return frames
    
    def _sample_context_and_reference_frames(self, frames):
        """
        Sample context and reference frames based on the sampling strategy
        
        Args:
            frames: List of frame tensors
            
        Returns:
            context_frames: Tensor of shape (context_size, 3, H, W)
            reference_frames: Tensor of shape (reference_size, 3, H, W)
        """
        total_frames = len(frames)
        min_required_frames = max(self.context_size, self.reference_size)
        
        if total_frames < min_required_frames:
            frames = frames * (min_required_frames // total_frames + 1)
            frames = frames[:min_required_frames]
            total_frames = len(frames)
        
        if self.sampling_strategy == 'temporal_split':
            ref_end_idx = min(self.reference_size, total_frames // 3)
            reference_indices = list(range(ref_end_idx))
            
            while len(reference_indices) < self.reference_size:
                reference_indices.append(reference_indices[-1])
            
            context_start = max(ref_end_idx, total_frames // 3)
            if total_frames - context_start >= self.context_size:
                context_indices = [
                    context_start + i * (total_frames - context_start - 1) // (self.context_size - 1)
                    for i in range(self.context_size)
                ]
            else:
                # Use all remaining frames and pad if necessary
                context_indices = list(range(context_start, total_frames))
                while len(context_indices) < self.context_size:
                    context_indices.append(context_indices[-1])
                    
        elif self.sampling_strategy == 'uniform':
            # Uniform sampling across all frames for both context and reference
            all_indices = list(range(total_frames))
            
            # Sample reference frames uniformly
            if self.reference_size <= total_frames:
                step = total_frames // self.reference_size
                reference_indices = [i * step for i in range(self.reference_size)]
            else:
                reference_indices = all_indices + [all_indices[-1]] * (self.reference_size - len(all_indices))
            
            # Sample context frames uniformly from remaining frames
            remaining_indices = [i for i in all_indices if i not in reference_indices[:self.reference_size]]
            if len(remaining_indices) >= self.context_size:
                step = len(remaining_indices) // self.context_size
                context_indices = [remaining_indices[i * step] for i in range(self.context_size)]
            else:
                context_indices = remaining_indices
                while len(context_indices) < self.context_size:
                    context_indices.append(context_indices[-1])
                    
        elif self.sampling_strategy == 'random':
            # Random sampling (different each time for training)
            all_indices = list(range(total_frames))
            
            if self.split == 'train':
                # Random sampling for training
                if total_frames >= self.reference_size + self.context_size:
                    sampled_indices = random.sample(all_indices, self.reference_size + self.context_size)
                    reference_indices = sorted(sampled_indices[:self.reference_size])
                    context_indices = sorted(sampled_indices[self.reference_size:])
                else:
                    reference_indices = random.choices(all_indices, k=self.reference_size)
                    context_indices = random.choices(all_indices, k=self.context_size)
            else:
                # Deterministic sampling for validation/test
                reference_indices = [i * total_frames // self.reference_size for i in range(self.reference_size)]
                context_indices = [(i + 1) * total_frames // (self.context_size + 1) for i in range(self.context_size)]
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        # Ensure indices are within bounds
        reference_indices = [min(i, total_frames - 1) for i in reference_indices]
        context_indices = [min(i, total_frames - 1) for i in context_indices]
        
        # Extract frames
        context_frames = torch.stack([frames[i] for i in context_indices], dim=0)
        reference_frames = torch.stack([frames[i] for i in reference_indices], dim=0)
        
        return context_frames, reference_frames
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames_folder = sample['frames_folder']
        label = sample['label']
        
        # Load all frames
        frames = self._load_frames(frames_folder)
        
        if len(frames) == 0:
            # Handle empty folders by creating dummy frames
            dummy_frame = torch.zeros(3, 224, 224)
            frames = [dummy_frame] * max(self.context_size, self.reference_size)
        
        context_frames, reference_frames = self._sample_context_and_reference_frames(frames)
        
        return {
            'context_frames': context_frames,
            'reference_frames': reference_frames,
            'label': label
        }

def create_dual_backbone_datasets(root_dir, context_size=5, reference_size=3, sampling_strategy='temporal_split'):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = EmbryoVideoDualBackboneDataset(
        root_dir=root_dir,
        split='train',
        transform=transform,
        max_frames=None,
        context_size=context_size,
        reference_size=reference_size,
        sampling_strategy=sampling_strategy
    )
    
    test_dataset = EmbryoVideoDualBackboneDataset(
        root_dir=root_dir,
        split='test',
        transform=transform,
        max_frames=None,
        context_size=context_size,
        reference_size=reference_size,
        sampling_strategy='uniform'
    )
    
    return train_dataset, test_dataset

def dual_backbone_collate_fn(batch):
    context_frames = torch.stack([item['context_frames'] for item in batch], dim=0)
    reference_frames = torch.stack([item['reference_frames'] for item in batch], dim=0)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    return context_frames, reference_frames, labels

if __name__ == "__main__":
    root_dir = 'D:/stroke/embryo/vin_embryov2/'
    
    train_dataset, test_dataset = create_dual_backbone_datasets(
        root_dir=root_dir,
        context_size=5,
        reference_size=3,
        sampling_strategy='temporal_split'
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    sample = train_dataset[0]
    print(f"Context frames shape: {sample['context_frames'].shape}")  # (5, 3, 224, 224)
    print(f"Reference frames shape: {sample['reference_frames'].shape}")  # (3, 3, 224, 224)
    print(f"Label: {sample['label']}")
    
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=dual_backbone_collate_fn
    )
    
    for context_frames, reference_frames, labels in train_loader:
        print(f"Batch context frames shape: {context_frames.shape}") # (4, 5, 3, 224, 224)
        print(f"Batch reference frames shape: {reference_frames.shape}")  # (4, 3, 3, 224, 224)
        print(f"Batch labels shape: {labels.shape}")  # (4,)
        break