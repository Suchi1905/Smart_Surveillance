"""
Hybrid Crash Classifier Service

Uses the trained YOLO + ViT + Fusion model for real-time crash classification.
Provides crash probability and confidence scores for video frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import timm
except ImportError:
    timm = None

logger = logging.getLogger(__name__)


class HybridClassifier:
    """
    Real-time crash classifier using YOLO + ViT hybrid model.
    
    Combines:
    - YOLO backbone features for spatial detection
    - MobileViT-S features for crash semantics
    - Cross-attention fusion for unified classification
    """
    
    def __init__(
        self,
        yolo_weights: Optional[str] = None,
        vit_weights: Optional[str] = None,
        fusion_weights: Optional[str] = None,
        e2e_weights: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        yolo_size: int = 640,
        vit_size: int = 256
    ):
        self.device = torch.device(device)
        self.yolo_size = yolo_size
        self.vit_size = vit_size
        self.is_loaded = False
        
        # Default weight paths
        base_path = Path(__file__).parent.parent.parent / "weights"
        
        self.yolo_weights = yolo_weights or str(base_path / "yolo_training" / "phase4_yolo" / "weights" / "best.pt")
        self.vit_weights = vit_weights or str(base_path / "vit_training" / "best_vit.pt")
        self.fusion_weights = fusion_weights or str(base_path / "fusion_training" / "best_fusion.pt")
        self.e2e_weights = e2e_weights or str(base_path / "e2e_training" / "best_e2e.pt")
        
        # Models (loaded lazily)
        self.yolo_model = None
        self.vit_backbone = None
        self.fusion = None
        self.yolo_proj = None
        
        logger.info(f"HybridClassifier initialized (device: {self.device})")
    
    def load(self) -> bool:
        """Load all model components."""
        if self.is_loaded:
            return True
        
        try:
            # 1. Load YOLO
            if YOLO is None:
                logger.error("ultralytics not installed")
                return False
            
            if Path(self.yolo_weights).exists():
                self.yolo = YOLO(self.yolo_weights)
                self.yolo_model = self.yolo.model.model
                self.yolo_model = self.yolo_model.to(self.device)
                self.yolo_model.eval()
                for param in self.yolo_model.parameters():
                    param.requires_grad = False
                logger.info("YOLO backbone loaded")
            else:
                logger.warning(f"YOLO weights not found: {self.yolo_weights}")
                return False
            
            # 2. Load ViT
            if timm is None:
                logger.error("timm not installed")
                return False
            
            self.vit_backbone = timm.create_model('mobilevit_s', pretrained=False, num_classes=0)
            
            # 3. Load Fusion layer (handle both package and standalone imports)
            try:
                from ..training.fusion_layer import CrossAttentionFusion
            except ImportError:
                import sys
                training_path = str(Path(__file__).parent.parent / "training")
                if training_path not in sys.path:
                    sys.path.insert(0, training_path)
                from fusion_layer import CrossAttentionFusion
            
            self.fusion = CrossAttentionFusion(
                yolo_dim=256,
                vit_dim=640,
                fusion_dim=256
            )
            
            # YOLO feature projection
            self.yolo_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 256)
            )
            
            # 4. Load weights - prefer E2E if available
            if Path(self.e2e_weights).exists():
                checkpoint = torch.load(self.e2e_weights, map_location='cpu', weights_only=False)
                self.vit_backbone.load_state_dict(checkpoint.get('vit_state_dict', {}), strict=False)
                self.fusion.load_state_dict(checkpoint.get('fusion_state_dict', {}), strict=False)
                self.yolo_proj.load_state_dict(checkpoint.get('yolo_proj_state_dict', {}), strict=False)
                logger.info("Loaded E2E weights")
            else:
                # Load individual weights
                if Path(self.vit_weights).exists():
                    vit_ckpt = torch.load(self.vit_weights, map_location='cpu', weights_only=False)
                    state_dict = vit_ckpt.get('model_state_dict', vit_ckpt)
                    backbone_state = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
                    if backbone_state:
                        self.vit_backbone.load_state_dict(backbone_state, strict=False)
                    logger.info("Loaded ViT weights")
                
                if Path(self.fusion_weights).exists():
                    fusion_ckpt = torch.load(self.fusion_weights, map_location='cpu', weights_only=False)
                    self.fusion.load_state_dict(fusion_ckpt.get('fusion_state_dict', {}), strict=False)
                    self.yolo_proj.load_state_dict(fusion_ckpt.get('yolo_proj_state_dict', {}), strict=False)
                    logger.info("Loaded fusion weights")
            
            # Move to device
            self.vit_backbone = self.vit_backbone.to(self.device)
            self.fusion = self.fusion.to(self.device)
            self.yolo_proj = self.yolo_proj.to(self.device)
            
            # Set eval mode
            self.vit_backbone.eval()
            self.fusion.eval()
            self.yolo_proj.eval()
            
            self.is_loaded = True
            logger.info("HybridClassifier fully loaded")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load HybridClassifier: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess frame for YOLO and ViT inputs.
        
        Args:
            frame: BGR frame from OpenCV
            
        Returns:
            Tuple of (yolo_tensor, vit_tensor)
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # YOLO input
        yolo_img = cv2.resize(rgb, (self.yolo_size, self.yolo_size))
        yolo_tensor = torch.from_numpy(yolo_img).permute(2, 0, 1).float() / 255.0
        yolo_tensor = yolo_tensor.unsqueeze(0).to(self.device)
        
        # ViT input with ImageNet normalization
        vit_img = cv2.resize(rgb, (self.vit_size, self.vit_size))
        vit_tensor = torch.from_numpy(vit_img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        vit_tensor = (vit_tensor.to(self.device) - mean) / std
        vit_tensor = vit_tensor.unsqueeze(0)
        
        return yolo_tensor, vit_tensor
    
    @torch.no_grad()
    def classify(self, frame: np.ndarray) -> Dict:
        """
        Classify frame for crash probability.
        
        Args:
            frame: BGR frame from OpenCV
            
        Returns:
            Dictionary with:
            - crash_probability: float (0-1)
            - is_crash: bool
            - confidence: float
            - inference_time_ms: float
        """
        if not self.is_loaded:
            if not self.load():
                return {
                    'crash_probability': 0.0,
                    'is_crash': False,
                    'confidence': 0.0,
                    'inference_time_ms': 0.0,
                    'error': 'Model not loaded'
                }
        
        import time
        start = time.time()
        
        # Preprocess
        yolo_tensor, vit_tensor = self.preprocess_frame(frame)
        
        # Extract YOLO features
        features = yolo_tensor
        for layer in self.yolo_model[:10]:
            features = layer(features)
        yolo_feat = self.yolo_proj(features)
        
        # Extract ViT features
        vit_feat = self.vit_backbone(vit_tensor)
        
        # Fusion and classification
        logits, _ = self.fusion(yolo_feat, vit_feat)
        probs = F.softmax(logits, dim=1)
        
        crash_prob = probs[0, 1].item()  # Class 1 = crash
        is_crash = crash_prob > 0.5
        confidence = probs.max().item()
        
        inference_time = (time.time() - start) * 1000
        
        return {
            'crash_probability': crash_prob,
            'is_crash': is_crash,
            'confidence': confidence,
            'inference_time_ms': inference_time
        }
    
    def get_info(self) -> Dict:
        """Get model information."""
        return {
            'is_loaded': self.is_loaded,
            'device': str(self.device),
            'yolo_weights': self.yolo_weights,
            'vit_weights': self.vit_weights,
            'fusion_weights': self.fusion_weights,
            'e2e_weights': self.e2e_weights
        }


# Singleton instance for easy import
_classifier_instance = None


def get_hybrid_classifier() -> HybridClassifier:
    """Get or create singleton HybridClassifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = HybridClassifier()
    return _classifier_instance


if __name__ == "__main__":
    # Test the classifier
    logging.basicConfig(level=logging.INFO)
    
    classifier = HybridClassifier()
    
    if classifier.load():
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = classifier.classify(test_frame)
        print(f"Classification result: {result}")
    else:
        print("Failed to load classifier")
