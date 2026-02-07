"""
Phase 4: Dataset Loaders for Hybrid Model Training

Provides PyTorch Dataset classes for:
- KITTI vehicle detection (YOLO training)
- IDD Indian driving (YOLO fine-tuning)  
- CCD crash videos (ViT classification)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random
import logging

logger = logging.getLogger(__name__)


class KITTIDataset(Dataset):
    """
    KITTI Object Detection Dataset for YOLO training.
    
    Converts KITTI format labels to YOLO format.
    """
    
    # KITTI to YOLO class mapping (vehicle-focused)
    KITTI_CLASSES = {
        'Car': 0,
        'Van': 1, 
        'Truck': 2,
        'Pedestrian': 3,
        'Person_sitting': 3,
        'Cyclist': 4,
        'Tram': 5,
        'Misc': 6,
        'DontCare': -1
    }
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        image_size: int = 640,
        augment: bool = True
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size
        self.augment = augment
        
        # Find all image files
        self.image_files = sorted(list(self.images_dir.glob("*.png")))
        if not self.image_files:
            self.image_files = sorted(list(self.images_dir.glob("**/*.png")))
        
        logger.info(f"KITTI Dataset: Found {len(self.image_files)} images")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def _parse_kitti_label(self, label_path: Path, img_w: int, img_h: int) -> List[List[float]]:
        """Convert KITTI label to YOLO format [class, x_center, y_center, width, height]"""
        labels = []
        
        if not label_path.exists():
            return labels
            
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                
                class_name = parts[0]
                if class_name not in self.KITTI_CLASSES or self.KITTI_CLASSES[class_name] == -1:
                    continue
                
                class_id = self.KITTI_CLASSES[class_name]
                
                # KITTI bbox: left, top, right, bottom (0-indexed)
                left = float(parts[4])
                top = float(parts[5])
                right = float(parts[6])
                bottom = float(parts[7])
                
                # Convert to YOLO format (normalized center coordinates)
                x_center = (left + right) / 2 / img_w
                y_center = (top + bottom) / 2 / img_h
                width = (right - left) / img_w
                height = (bottom - top) / img_h
                
                # Clamp to valid range
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                labels.append([class_id, x_center, y_center, width, height])
        
        return labels
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Parse labels
        labels = self._parse_kitti_label(label_path, w, h)
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Augmentation
        if self.augment:
            image = self._augment(image)
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
        
        return image, labels
    
    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Basic augmentation"""
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        return image


class CrashVideoDataset(Dataset):
    """
    CCD Crash Video Dataset for ViT classification.
    
    Samples frames from crash/normal videos for binary classification.
    """
    
    def __init__(
        self,
        crash_dir: str,
        normal_dir: str,
        frames_per_video: int = 8,
        image_size: int = 256,
        augment: bool = True,
        max_videos: Optional[int] = None
    ):
        self.crash_dir = Path(crash_dir)
        self.normal_dir = Path(normal_dir)
        self.frames_per_video = frames_per_video
        self.image_size = image_size
        self.augment = augment
        
        # Collect video paths
        self.samples = []
        
        # Crash videos (label = 1)
        crash_videos = sorted(list(self.crash_dir.glob("*.mp4")))
        if max_videos:
            crash_videos = crash_videos[:max_videos]
        for vid in crash_videos:
            self.samples.append((vid, 1))
        
        # Normal videos (label = 0)
        normal_videos = sorted(list(self.normal_dir.glob("*.mp4")))
        if max_videos:
            normal_videos = normal_videos[:max_videos]
        for vid in normal_videos:
            self.samples.append((vid, 0))
        
        logger.info(f"CrashVideoDataset: {len(crash_videos)} crash, {len(normal_videos)} normal videos")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _extract_frames(self, video_path: Path) -> List[np.ndarray]:
        """Extract evenly spaced frames from video"""
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)] * self.frames_per_video
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 50  # Default for CCD videos
        
        # Calculate frame indices to sample
        indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frames.append(frame)
            else:
                frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        
        cap.release()
        return frames
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        
        # Extract frames
        frames = self._extract_frames(video_path)
        
        # Augmentation
        if self.augment:
            frames = [self._augment(f) for f in frames]
        
        # Stack frames: [frames_per_video, C, H, W]
        frames_tensor = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            for f in frames
        ])
        
        return frames_tensor, label
    
    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Basic augmentation for frames"""
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        return image


class IDDDataset(Dataset):
    """
    Indian Driving Dataset for YOLO fine-tuning.
    
    Focuses on Indian traffic patterns and vehicle types.
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 640,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment
        
        # Find all images
        self.image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            self.image_files.extend(list(self.data_dir.glob(f"**/{ext}")))
        
        self.image_files = sorted(self.image_files)
        logger.info(f"IDD Dataset: Found {len(self.image_files)} images")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.image_files[idx]
        
        # Load and resize image
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.image_size, self.image_size))
        
        if self.augment:
            image = self._augment(image)
        
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, str(img_path)
    
    def _augment(self, image: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        return image


def create_dataloaders(config) -> Dict[str, DataLoader]:
    """Create dataloaders for all datasets"""
    from .hybrid_config import config as default_config
    cfg = config or default_config
    
    dataloaders = {}
    
    # KITTI dataloader
    if cfg.dataset.kitti_images_dir.exists():
        kitti_dataset = KITTIDataset(
            images_dir=str(cfg.dataset.kitti_images_dir),
            labels_dir=str(cfg.dataset.kitti_labels_dir),
            image_size=cfg.yolo.image_size
        )
        dataloaders['kitti'] = DataLoader(
            kitti_dataset,
            batch_size=cfg.yolo.batch_size,
            shuffle=True,
            num_workers=cfg.yolo.workers,
            pin_memory=True
        )
    
    # CCD Crash dataloader
    if cfg.dataset.ccd_crash_dir.exists():
        crash_dataset = CrashVideoDataset(
            crash_dir=str(cfg.dataset.ccd_crash_dir),
            normal_dir=str(cfg.dataset.ccd_normal_dir),
            frames_per_video=cfg.vit.frames_per_video,
            image_size=cfg.vit.image_size
        )
        dataloaders['crash'] = DataLoader(
            crash_dataset,
            batch_size=cfg.vit.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    return dataloaders


if __name__ == "__main__":
    # Test dataset loading
    logging.basicConfig(level=logging.INFO)
    
    from hybrid_config import config
    
    print("Testing KITTI Dataset...")
    if config.dataset.kitti_labels_dir.exists():
        kitti = KITTIDataset(
            str(config.dataset.kitti_images_dir),
            str(config.dataset.kitti_labels_dir)
        )
        print(f"  Loaded {len(kitti)} samples")
        if len(kitti) > 0:
            img, labels = kitti[0]
            print(f"  Image shape: {img.shape}, Labels shape: {labels.shape}")
    
    print("\nTesting CCD Crash Dataset...")
    if config.dataset.ccd_crash_dir.exists():
        crash = CrashVideoDataset(
            str(config.dataset.ccd_crash_dir),
            str(config.dataset.ccd_normal_dir),
            max_videos=5
        )
        print(f"  Loaded {len(crash)} samples")
        if len(crash) > 0:
            frames, label = crash[0]
            print(f"  Frames shape: {frames.shape}, Label: {label}")
