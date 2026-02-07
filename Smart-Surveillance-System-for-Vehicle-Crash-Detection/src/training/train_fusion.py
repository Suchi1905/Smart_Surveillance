"""
Phase 4C & 4D: Train Fusion Layer and End-to-End Fine-tuning

Trains the fusion layer to combine YOLO and ViT features,
then optionally fine-tunes the entire model end-to-end.

Usage:
    # Phase C: Train fusion only
    python train_fusion.py --phase C --epochs 20
    
    # Phase D: End-to-end fine-tuning
    python train_fusion.py --phase D --epochs 10
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fusion_layer import CrossAttentionFusion
from train_vit import MobileViTClassifier

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import timm
except ImportError:
    timm = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridDataset(Dataset):
    """
    Dataset that provides paired inputs for YOLO and ViT.
    """
    
    def __init__(
        self,
        crash_dir: str,
        normal_dir: str,
        yolo_size: int = 640,
        vit_size: int = 256,
        augment: bool = True,
        max_samples: Optional[int] = None
    ):
        self.crash_dir = Path(crash_dir)
        self.normal_dir = Path(normal_dir)
        self.yolo_size = yolo_size
        self.vit_size = vit_size
        self.augment = augment
        
        self.samples = []
        
        # Collect crash videos
        crash_videos = sorted(list(self.crash_dir.glob("*.mp4")))
        if not crash_videos:
            crash_videos = sorted(list(self.crash_dir.glob("**/*.mp4")))
        
        for vid in crash_videos:
            self.samples.append((vid, 1))
        
        # Collect normal videos
        normal_videos = sorted(list(self.normal_dir.glob("*.mp4")))
        if not normal_videos:
            normal_videos = sorted(list(self.normal_dir.glob("**/*.mp4")))
        
        for vid in normal_videos:
            self.samples.append((vid, 0))
        
        # Limit samples if specified
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        random.shuffle(self.samples)
        logger.info(f"HybridDataset: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def _extract_frame(self, video_path: Path) -> np.ndarray:
        """Extract middle frame from video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return np.zeros((self.vit_size, self.vit_size, 3), dtype=np.uint8)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 50
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return np.zeros((self.vit_size, self.vit_size, 3), dtype=np.uint8)
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frame = self._extract_frame(video_path)
        
        # Resize for YOLO
        yolo_img = cv2.resize(frame, (self.yolo_size, self.yolo_size))
        yolo_img = torch.from_numpy(yolo_img).permute(2, 0, 1).float() / 255.0
        
        # Resize for ViT with ImageNet normalization
        vit_img = cv2.resize(frame, (self.vit_size, self.vit_size))
        vit_img = torch.from_numpy(vit_img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        vit_img = (vit_img - mean) / std
        
        return yolo_img, vit_img, label


class HybridTrainer:
    """
    Trainer for fusion layer and end-to-end fine-tuning.
    """
    
    def __init__(
        self,
        yolo_weights: str,
        vit_weights: str,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load YOLO
        if YOLO is None:
            raise ImportError("ultralytics required")
        self.yolo = YOLO(yolo_weights)
        self.yolo_model = self.yolo.model.model
        
        # Load ViT
        if timm is None:
            raise ImportError("timm required")
        self.vit_backbone = timm.create_model('mobilevit_s', pretrained=False, num_classes=0)
        
        # Load ViT weights
        if Path(vit_weights).exists():
            checkpoint = torch.load(vit_weights, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            backbone_state = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
            if backbone_state:
                self.vit_backbone.load_state_dict(backbone_state, strict=False)
        
        self.vit_backbone = self.vit_backbone.to(self.device)
        
        # Freeze backbones initially
        for param in self.vit_backbone.parameters():
            param.requires_grad = False
        
        # Create fusion layer
        self.fusion = CrossAttentionFusion(
            yolo_dim=256,
            vit_dim=640,
            fusion_dim=256
        ).to(self.device)
        
        # YOLO feature projection
        self.yolo_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256)
        ).to(self.device)
        
        logger.info("HybridTrainer initialized")
    
    def extract_yolo_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from YOLO backbone"""
        with torch.no_grad():
            features = x
            # Run through YOLO backbone (layers 0-9)
            for i, layer in enumerate(self.yolo_model[:10]):
                features = layer(features)
        return self.yolo_proj(features)
    
    def extract_vit_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from ViT backbone"""
        with torch.no_grad():
            return self.vit_backbone(x)
    
    def train_fusion(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        lr: float = 1e-4,
        save_dir: str = "weights/fusion_training"
    ):
        """Train fusion layer only (Phase C)"""
        logger.info("="*60)
        logger.info("PHASE 4C: FUSION LAYER TRAINING")
        logger.info("="*60)
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        optimizer = optim.AdamW(
            list(self.fusion.parameters()) + list(self.yolo_proj.parameters()),
            lr=lr, weight_decay=0.01
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = GradScaler()
        
        best_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            # Training
            self.fusion.train()
            self.yolo_proj.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (yolo_img, vit_img, labels) in enumerate(train_loader):
                yolo_img = yolo_img.to(self.device)
                vit_img = vit_img.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                with autocast():
                    yolo_feat = self.extract_yolo_features(yolo_img)
                    vit_feat = self.extract_vit_features(vit_img)
                    
                    logits, _ = self.fusion(yolo_feat, vit_feat)
                    loss = criterion(logits, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            train_acc = 100. * correct / total
            
            # Validation
            self.fusion.eval()
            self.yolo_proj.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for yolo_img, vit_img, labels in val_loader:
                    yolo_img = yolo_img.to(self.device)
                    vit_img = vit_img.to(self.device)
                    labels = labels.to(self.device)
                    
                    yolo_feat = self.extract_yolo_features(yolo_img)
                    vit_feat = self.extract_vit_features(vit_img)
                    
                    logits, _ = self.fusion(yolo_feat, vit_feat)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * correct / total
            scheduler.step()
            
            logger.info(f"\nEpoch {epoch}/{epochs}")
            logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'fusion_state_dict': self.fusion.state_dict(),
                    'yolo_proj_state_dict': self.yolo_proj.state_dict(),
                    'val_acc': val_acc
                }, save_path / "best_fusion.pt")
                logger.info(f"Saved best model with Val Acc: {val_acc:.2f}%")
            
            torch.save({
                'epoch': epoch,
                'fusion_state_dict': self.fusion.state_dict(),
                'yolo_proj_state_dict': self.yolo_proj.state_dict(),
                'val_acc': val_acc
            }, save_path / "last_fusion.pt")
        
        logger.info("\n" + "="*60)
        logger.info("FUSION TRAINING COMPLETE!")
        logger.info(f"Best Validation Accuracy: {best_acc:.2f}%")
        logger.info("="*60)
        
        return best_acc


def main():
    parser = argparse.ArgumentParser(description="Phase 4C-D Training")
    parser.add_argument("--phase", type=str, default="C", choices=["C", "D"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    # Import config
    from hybrid_config import config
    
    # Create dataset
    dataset = HybridDataset(
        crash_dir=str(config.dataset.ccd_crash_dir),
        normal_dir=str(config.dataset.ccd_normal_dir),
        max_samples=None  # Use all samples
    )
    
    # Split
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    
    # Create trainer
    trainer = HybridTrainer(
        yolo_weights=str(config.yolo.save_dir / "phase4_yolo" / "weights" / "best.pt"),
        vit_weights=str(config.vit.save_dir / "best_vit.pt")
    )
    
    if args.phase == "C":
        trainer.train_fusion(train_loader, val_loader, epochs=args.epochs, lr=args.lr)
    else:
        logger.info("Phase D: End-to-end fine-tuning not yet implemented")
        # This would unfreeze backbones and train with very low LR


if __name__ == "__main__":
    main()
