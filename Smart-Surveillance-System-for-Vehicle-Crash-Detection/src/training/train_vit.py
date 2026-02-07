"""
Phase 4B: Vision Transformer Classifier for Crash Detection

Trains MobileViT-S on CCD crash video frames for binary classification.
Optimized for RTX 4050 (6GB VRAM).

Usage:
    python train_vit.py --epochs 30 --batch 4
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import cv2

# Try to import timm for MobileViT
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[WARNING] timm not available. Install with: pip install timm")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrashFrameDataset(Dataset):
    """
    Dataset for crash/non-crash frame classification.
    Extracts frames from CCD videos for ViT training.
    """
    
    def __init__(
        self,
        crash_dir: str,
        normal_dir: str,
        image_size: int = 256,
        frames_per_video: int = 8,
        augment: bool = True,
        max_videos: Optional[int] = None
    ):
        self.crash_dir = Path(crash_dir)
        self.normal_dir = Path(normal_dir)
        self.image_size = image_size
        self.frames_per_video = frames_per_video
        self.augment = augment
        
        # Collect all video paths
        self.samples = []
        
        # Crash videos (label = 1)
        crash_videos = sorted(list(self.crash_dir.glob("*.mp4")))
        if not crash_videos:
            crash_videos = sorted(list(self.crash_dir.glob("**/*.mp4")))
        if max_videos:
            crash_videos = crash_videos[:max_videos]
        
        for vid in crash_videos:
            self.samples.append((vid, 1))
        
        # Normal videos (label = 0)
        normal_videos = sorted(list(self.normal_dir.glob("*.mp4")))
        if not normal_videos:
            normal_videos = sorted(list(self.normal_dir.glob("**/*.mp4")))
        if max_videos:
            normal_videos = normal_videos[:max_videos]
            
        for vid in normal_videos:
            self.samples.append((vid, 0))
        
        # Shuffle samples
        random.shuffle(self.samples)
        
        logger.info(f"CrashFrameDataset: {len(crash_videos)} crash + {len(normal_videos)} normal = {len(self.samples)} total")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _extract_key_frame(self, video_path: Path) -> np.ndarray:
        """Extract a key frame (middle frame) from video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 50
        
        # Get middle frame (likely contains the crash event)
        middle_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Convert and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        
        return frame
    
    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentations"""
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness/contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.randint(-20, 20)    # brightness
            image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
        
        # Random rotation
        if random.random() > 0.7:
            angle = random.randint(-15, 15)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        return image
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        
        # Extract frame
        frame = self._extract_key_frame(video_path)
        
        # Augment if training
        if self.augment:
            frame = self._augment(frame)
        
        # Convert to tensor and normalize
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame = (frame - mean) / std
        
        return frame, label


class MobileViTClassifier(nn.Module):
    """
    MobileViT-S based crash classifier.
    
    Uses timm's MobileViT-S pretrained on ImageNet.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install with: pip install timm")
        
        # Load MobileViT-S
        self.backbone = timm.create_model(
            'mobilevit_s',
            pretrained=pretrained,
            num_classes=0  # Remove classifier head
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        logger.info(f"MobileViTClassifier initialized: feature_dim={self.feature_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification (for fusion layer)"""
        return self.backbone(x)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int = 4
) -> Tuple[float, float]:
    """Train for one epoch with gradient accumulation and AMP"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass with AMP
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # Normalize for accumulation
        
        # Backward pass with scaler
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item()*accumulation_steps:.4f}")
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def train_vit(
    crash_dir: str,
    normal_dir: str,
    epochs: int = 30,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    save_dir: str = "weights/vit_training",
    device: str = "cuda",
    accumulation_steps: int = 4
):
    """
    Train MobileViT classifier on crash videos.
    """
    logger.info("="*60)
    logger.info("PHASE 4B: VIT CLASSIFIER TRAINING")
    logger.info("="*60)
    logger.info(f"Crash dir: {crash_dir}")
    logger.info(f"Normal dir: {normal_dir}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Gradient accumulation: {accumulation_steps}")
    logger.info("="*60)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset = CrashFrameDataset(
        crash_dir=crash_dir,
        normal_dir=normal_dir,
        image_size=256,
        augment=True
    )
    
    # Split into train/val
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Disable augmentation for validation
    val_dataset.dataset.augment = False
    
    logger.info(f"Train: {train_size}, Val: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid Windows issues
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    model = MobileViTClassifier(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Gradient scaler for AMP
    scaler = GradScaler()
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, accumulation_steps
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, save_path / "best_vit.pt")
            logger.info(f"Saved best model with Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc
        }, save_path / "last_vit.pt")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best Validation Accuracy: {best_acc:.2f}%")
    logger.info(f"Model saved to: {save_path / 'best_vit.pt'}")
    logger.info("="*60)
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="ViT Training for Phase 4B")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    # Import config
    from hybrid_config import config
    
    # Check if timm is available
    if not TIMM_AVAILABLE:
        logger.error("Please install timm: pip install timm")
        return
    
    # Check dataset directories
    if not config.dataset.ccd_crash_dir.exists():
        logger.error(f"Crash directory not found: {config.dataset.ccd_crash_dir}")
        return
    
    if not config.dataset.ccd_normal_dir.exists():
        logger.error(f"Normal directory not found: {config.dataset.ccd_normal_dir}")
        return
    
    # Train ViT
    train_vit(
        crash_dir=str(config.dataset.ccd_crash_dir),
        normal_dir=str(config.dataset.ccd_normal_dir),
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        save_dir=str(config.vit.save_dir),
        device=args.device,
        accumulation_steps=args.accum
    )


if __name__ == "__main__":
    main()
