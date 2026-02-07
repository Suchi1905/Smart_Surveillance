"""
Phase 4: YOLO + ViT Hybrid Model Training Configuration

Optimized for RTX 4050 (6GB VRAM), Ryzen 7, 24GB RAM
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "Datasets"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

@dataclass
class DatasetConfig:
    """Dataset paths and configuration"""
    
    # CCD Crash Dataset
    ccd_crash_dir: Path = DATASETS_DIR / "videos-20260206T085121Z-1-002" / "videos" / "Crash-1500"
    ccd_normal_dir: Path = DATASETS_DIR / "Normal-001"
    ccd_annotations: Path = DATASETS_DIR / "videos-20260206T085121Z-1-002" / "videos" / "Crash-1500.txt"
    
    # KITTI Dataset
    kitti_images_dir: Path = DATASETS_DIR / "data_object_image_2"
    kitti_labels_dir: Path = DATASETS_DIR / "data_object_label_2"
    
    # IDD (Indian Driving Dataset)
    idd_dir: Path = DATASETS_DIR / "idd-20k-II" / "idd20kII"
    
    # DoTA (Detection of Traffic Anomaly)
    dota_dir: Path = DATASETS_DIR / "Detection-of-Traffic-Anomaly-master"
    
    # Train/Val/Test splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class YOLOConfig:
    """YOLO backbone training configuration"""
    
    # Model
    base_model: str = "yolov8n.pt"  # Nano for RTX 4050
    
    # Training params (optimized for 6GB VRAM)
    batch_size: int = 4
    gradient_accumulation: int = 4  # Effective batch = 16
    epochs: int = 50
    image_size: int = 640
    
    # Learning rate
    lr0: float = 0.01
    lrf: float = 0.01
    warmup_epochs: int = 3
    
    # Augmentation
    augment: bool = True
    mosaic: float = 0.5
    mixup: float = 0.0
    
    # Hardware
    device: str = "0"  # GPU 0
    workers: int = 4
    
    # Output
    project: str = str(WEIGHTS_DIR / "yolo_training")
    name: str = "phase4_yolo"


@dataclass
class ViTConfig:
    """Vision Transformer configuration"""
    
    # Model - MobileViT-S for memory efficiency
    model_name: str = "mobilevit_s"
    pretrained: bool = True
    num_classes: int = 2  # Crash / Non-Crash
    
    # Training params (optimized for 6GB VRAM)
    batch_size: int = 4
    gradient_accumulation: int = 4  # Effective batch = 16
    epochs: int = 30
    image_size: int = 256  # ViT input size
    
    # Learning rate
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Mixed precision
    use_amp: bool = True  # FP16 for memory
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    
    # Video processing
    frames_per_video: int = 8  # Sample 8 frames per video
    
    # Output
    save_dir: Path = WEIGHTS_DIR / "vit_training"


@dataclass 
class FusionConfig:
    """Fusion layer configuration"""
    
    # Architecture
    yolo_feature_dim: int = 256
    vit_feature_dim: int = 384  # MobileViT-S output
    fusion_dim: int = 256
    num_heads: int = 4
    
    # Training
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 1e-4
    
    # Freeze backbones
    freeze_yolo: bool = True
    freeze_vit: bool = True
    
    # Output
    save_dir: Path = WEIGHTS_DIR / "fusion_training"


@dataclass
class HybridConfig:
    """Complete hybrid model configuration"""
    
    # Component configs
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    vit: ViTConfig = field(default_factory=ViTConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    
    # Paths
    yolo_weights: Path = WEIGHTS_DIR / "yolo_training" / "phase4_yolo" / "weights" / "best.pt"
    vit_weights: Path = WEIGHTS_DIR / "vit_training" / "best_vit.pt"
    fusion_weights: Path = WEIGHTS_DIR / "fusion_training" / "best_fusion.pt"
    hybrid_weights: Path = WEIGHTS_DIR / "hybrid_model.pt"
    
    # Inference
    confidence_threshold: float = 0.5
    use_hybrid: bool = True
    
    def ensure_dirs(self):
        """Create necessary directories"""
        WEIGHTS_DIR.mkdir(exist_ok=True)
        self.vit.save_dir.mkdir(parents=True, exist_ok=True)
        self.fusion.save_dir.mkdir(parents=True, exist_ok=True)


# Default configuration instance
config = HybridConfig()
