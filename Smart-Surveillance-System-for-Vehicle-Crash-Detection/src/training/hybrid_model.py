"""
Phase 4: Complete YOLO + ViT Hybrid Model

Combines trained YOLO backbone, ViT classifier, and fusion layer
into a unified crash detection model.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import timm
except ImportError:
    timm = None

from .fusion_layer import CrossAttentionFusion, SimpleFusion

logger = logging.getLogger(__name__)


class YOLOFeatureExtractor(nn.Module):
    """
    Extract features from YOLO backbone.
    Uses the last feature layer before detection head.
    """
    
    def __init__(self, yolo_weights: str, output_dim: int = 256):
        super().__init__()
        
        if YOLO is None:
            raise ImportError("ultralytics is required")
        
        # Load YOLO model
        self.yolo = YOLO(yolo_weights)
        self.yolo_model = self.yolo.model
        
        # Freeze YOLO backbone
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        
        # Feature projection (YOLO P5 features are 256 channels for yolov8n)
        self.feature_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, output_dim)
        )
        
        logger.info(f"YOLOFeatureExtractor loaded from {yolo_weights}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from YOLO backbone"""
        # Get intermediate features from YOLO backbone
        # YOLOv8n backbone outputs at indices 4, 6, 9 (P3, P4, P5)
        with torch.no_grad():
            # Run through backbone layers
            features = x
            for i, layer in enumerate(self.yolo_model.model[:10]):  # Up to SPPF
                features = layer(features)
        
        # Project to output dimension
        features = self.feature_proj(features)
        return features


class ViTFeatureExtractor(nn.Module):
    """
    Extract features from trained MobileViT classifier.
    """
    
    def __init__(self, vit_weights: str, output_dim: int = 640):
        super().__init__()
        
        if timm is None:
            raise ImportError("timm is required")
        
        # Load MobileViT backbone
        self.backbone = timm.create_model(
            'mobilevit_s',
            pretrained=False,
            num_classes=0
        )
        
        # Load trained weights
        if Path(vit_weights).exists():
            checkpoint = torch.load(vit_weights, map_location='cpu')
            
            # Extract backbone weights from full model
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            backbone_state = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    backbone_state[k.replace('backbone.', '')] = v
            
            if backbone_state:
                self.backbone.load_state_dict(backbone_state, strict=False)
                logger.info(f"ViT backbone loaded from {vit_weights}")
            else:
                logger.warning(f"Could not load backbone weights from {vit_weights}")
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from ViT backbone"""
        with torch.no_grad():
            features = self.backbone(x)
        return features


class HybridCrashDetector(nn.Module):
    """
    Complete Hybrid Model combining YOLO + ViT + Fusion.
    
    Architecture:
        1. YOLO backbone: Extracts spatial features for detection
        2. ViT backbone: Extracts semantic features for classification
        3. Fusion layer: Combines both using cross-attention
        4. Classifier: Final crash/non-crash prediction
    """
    
    def __init__(
        self,
        yolo_weights: str,
        vit_weights: str,
        fusion_weights: Optional[str] = None,
        yolo_dim: int = 256,
        vit_dim: int = 640,
        fusion_dim: int = 256,
        use_cross_attention: bool = True
    ):
        super().__init__()
        
        # Initialize feature extractors
        self.yolo_extractor = YOLOFeatureExtractor(yolo_weights, yolo_dim)
        self.vit_extractor = ViTFeatureExtractor(vit_weights, vit_dim)
        
        # Initialize fusion layer
        if use_cross_attention:
            self.fusion = CrossAttentionFusion(
                yolo_dim=yolo_dim,
                vit_dim=vit_dim,
                fusion_dim=fusion_dim
            )
        else:
            self.fusion = SimpleFusion(
                yolo_dim=yolo_dim,
                vit_dim=vit_dim,
                fusion_dim=fusion_dim
            )
        
        # Load fusion weights if provided
        if fusion_weights and Path(fusion_weights).exists():
            fusion_state = torch.load(fusion_weights, map_location='cpu')
            self.fusion.load_state_dict(fusion_state.get('model_state_dict', fusion_state))
            logger.info(f"Fusion layer loaded from {fusion_weights}")
        
        logger.info("HybridCrashDetector initialized")
    
    def forward(
        self,
        yolo_input: torch.Tensor,
        vit_input: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid model.
        
        Args:
            yolo_input: Image for YOLO [B, 3, 640, 640]
            vit_input: Image for ViT [B, 3, 256, 256]
            
        Returns:
            Dict with 'logits', 'probs', 'prediction', 'fused_features'
        """
        # Extract features
        yolo_features = self.yolo_extractor(yolo_input)
        vit_features = self.vit_extractor(vit_input)
        
        # Fuse features
        logits, fused_features = self.fusion(yolo_features, vit_features)
        
        # Compute probabilities and prediction
        probs = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probs': probs,
            'prediction': prediction,
            'crash_prob': probs[:, 1],  # Probability of crash
            'fused_features': fused_features
        }
    
    def predict(self, image: torch.Tensor) -> Dict[str, Any]:
        """
        Single image prediction helper.
        
        Args:
            image: Input image [3, H, W] or [B, 3, H, W]
            
        Returns:
            Prediction dictionary
        """
        import torch.nn.functional as F
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Resize for YOLO and ViT
        yolo_input = F.interpolate(image, size=(640, 640), mode='bilinear')
        vit_input = F.interpolate(image, size=(256, 256), mode='bilinear')
        
        with torch.no_grad():
            return self(yolo_input, vit_input)
    
    def freeze_backbones(self):
        """Freeze YOLO and ViT backbones for fusion training"""
        for param in self.yolo_extractor.parameters():
            param.requires_grad = False
        for param in self.vit_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters for end-to-end fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True


def load_hybrid_model(
    yolo_weights: str = "weights/yolo_training/phase4_yolo/weights/best.pt",
    vit_weights: str = "weights/vit_training/best_vit.pt",
    fusion_weights: Optional[str] = "weights/fusion_training/best_fusion.pt",
    device: str = "cuda"
) -> HybridCrashDetector:
    """
    Load complete hybrid model with trained weights.
    """
    model = HybridCrashDetector(
        yolo_weights=yolo_weights,
        vit_weights=vit_weights,
        fusion_weights=fusion_weights
    )
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with dummy inputs
    print("Testing HybridCrashDetector...")
    
    # Create dummy model (without loading actual weights)
    yolo_feat = torch.randn(2, 256)
    vit_feat = torch.randn(2, 640)
    
    fusion = CrossAttentionFusion()
    logits, fused = fusion(yolo_feat, vit_feat)
    
    print(f"Fusion output - logits: {logits.shape}, fused: {fused.shape}")
    print(f"Predictions: {torch.softmax(logits, dim=-1)}")
