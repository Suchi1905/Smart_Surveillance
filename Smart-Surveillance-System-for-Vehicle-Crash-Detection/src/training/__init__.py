"""
Training module for Phase 4: YOLO + ViT Hybrid Model

This module contains:
- hybrid_config: Configuration for all training phases
- dataset_loader: PyTorch Datasets for KITTI, CCD, IDD
- train_yolo: Phase A YOLO backbone training
- train_vit: Phase B ViT classifier training
- fusion_layer: Cross-attention fusion module
- hybrid_model: Complete hybrid crash detector
- train_fusion: Phase C-D fusion and end-to-end training
"""

from .hybrid_config import config, HybridConfig
from .fusion_layer import CrossAttentionFusion, SimpleFusion

__all__ = [
    'config',
    'HybridConfig',
    'create_dataloaders'
]
