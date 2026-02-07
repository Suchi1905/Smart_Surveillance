"""
Phase 4C: Cross-Attention Fusion Layer

Combines YOLO detection features with ViT classification features
using a cross-attention mechanism for improved crash detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion module that combines YOLO and ViT features.
    
    Uses multi-head attention to let YOLO features attend to ViT features
    and vice versa, producing a unified representation.
    """
    
    def __init__(
        self,
        yolo_dim: int = 256,
        vit_dim: int = 640,  # MobileViT-S output dimension
        fusion_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.yolo_dim = yolo_dim
        self.vit_dim = vit_dim
        self.fusion_dim = fusion_dim
        
        # Project YOLO and ViT features to same dimension
        self.yolo_proj = nn.Linear(yolo_dim, fusion_dim)
        self.vit_proj = nn.Linear(vit_dim, fusion_dim)
        
        # Cross-attention: YOLO attends to ViT
        self.yolo_to_vit_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention: ViT attends to YOLO
        self.vit_to_yolo_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        
        # Feed-forward fusion
        self.fusion_ffn = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Crash / Non-crash
        )
        
        logger.info(f"CrossAttentionFusion: yolo={yolo_dim}, vit={vit_dim}, fusion={fusion_dim}")
    
    def forward(
        self,
        yolo_features: torch.Tensor,
        vit_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for fusion.
        
        Args:
            yolo_features: YOLO backbone features [B, yolo_dim] or [B, N, yolo_dim]
            vit_features: ViT features [B, vit_dim]
            
        Returns:
            Tuple of (crash_logits, fused_features)
        """
        # Ensure features are 3D for attention [B, seq_len, dim]
        if yolo_features.dim() == 2:
            yolo_features = yolo_features.unsqueeze(1)  # [B, 1, yolo_dim]
        
        if vit_features.dim() == 2:
            vit_features = vit_features.unsqueeze(1)  # [B, 1, vit_dim]
        
        # Project to fusion dimension
        yolo_proj = self.yolo_proj(yolo_features)  # [B, 1, fusion_dim]
        vit_proj = self.vit_proj(vit_features)      # [B, 1, fusion_dim]
        
        # Cross-attention: YOLO attends to ViT
        yolo_attended, _ = self.yolo_to_vit_attn(
            query=yolo_proj,
            key=vit_proj,
            value=vit_proj
        )
        yolo_attended = self.norm1(yolo_proj + yolo_attended)
        
        # Cross-attention: ViT attends to YOLO
        vit_attended, _ = self.vit_to_yolo_attn(
            query=vit_proj,
            key=yolo_proj,
            value=yolo_proj
        )
        vit_attended = self.norm2(vit_proj + vit_attended)
        
        # Concatenate and fuse
        combined = torch.cat([yolo_attended, vit_attended], dim=-1)  # [B, 1, fusion_dim*2]
        fused = self.fusion_ffn(combined)  # [B, 1, fusion_dim]
        
        # Squeeze sequence dimension
        fused = fused.squeeze(1)  # [B, fusion_dim]
        
        # Classify
        logits = self.classifier(fused)  # [B, 2]
        
        return logits, fused


class SimpleFusion(nn.Module):
    """
    Simple concatenation-based fusion (lighter alternative).
    """
    
    def __init__(
        self,
        yolo_dim: int = 256,
        vit_dim: int = 640,
        fusion_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(yolo_dim + vit_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(fusion_dim, 2)
        
    def forward(
        self,
        yolo_features: torch.Tensor,
        vit_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Flatten if needed
        if yolo_features.dim() > 2:
            yolo_features = yolo_features.view(yolo_features.size(0), -1)
        if vit_features.dim() > 2:
            vit_features = vit_features.view(vit_features.size(0), -1)
        
        # Concatenate and fuse
        combined = torch.cat([yolo_features, vit_features], dim=-1)
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        
        return logits, fused


if __name__ == "__main__":
    # Test fusion module
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy features
    batch_size = 4
    yolo_feat = torch.randn(batch_size, 256)
    vit_feat = torch.randn(batch_size, 640)
    
    # Test CrossAttentionFusion
    fusion = CrossAttentionFusion()
    logits, fused = fusion(yolo_feat, vit_feat)
    print(f"CrossAttention - Logits: {logits.shape}, Fused: {fused.shape}")
    
    # Test SimpleFusion
    simple = SimpleFusion()
    logits2, fused2 = simple(yolo_feat, vit_feat)
    print(f"SimpleFusion - Logits: {logits2.shape}, Fused: {fused2.shape}")
