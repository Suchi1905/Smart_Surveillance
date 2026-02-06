# Phase 4: YOLO + Vision Transformer Hybrid Implementation

## Overview

Implement a hybrid detection model that combines YOLO's fast object detection with Vision Transformer's superior crash classification accuracy. This enhances the existing crash detection pipeline.

## Dataset Verification ✅

All required datasets are present in `Datasets/`:

| Dataset | Location | Size | Purpose |
|---------|----------|------|---------|
| **CCD Crash Videos** | `videos-20260206T085121Z-1-002/videos/Crash-1500/` | 1,500 videos | ViT crash classification |
| **Normal Videos** | `Normal-001/` | 3,000 videos | ViT negative samples |
| **KITTI Labels** | `data_object_label_2/` + `kitti_labels/` | 7,481 labels | YOLO backbone training |
| **IDD (Indian)** | `idd-20k-II/` | 8,089 images | Indian traffic fine-tuning |
| **DoTA Anomaly** | `Detection-of-Traffic-Anomaly-master/` | Various | Anomaly detection |
| **US Accidents** | `archive (13)/US_Accidents_March23.csv` | 3GB CSV | Statistical analysis |

---

## User Review Required

> [!IMPORTANT]
> **Hardware Constraints (RTX 4050, 6GB VRAM):**
> - Training will use batch size 4 with gradient accumulation
> - Mixed precision (FP16) enabled for memory efficiency
> - MobileViT-S selected as ViT variant (25MB, fits in VRAM)

> [!WARNING]
> **Training Time Estimate:**
> - Phase A (YOLO backbone): ~6-8 hours
> - Phase B (ViT classifier): ~4-6 hours
> - Phase C (Fusion layer): ~2-3 hours
> - Phase D (End-to-end): ~2-3 hours
> - **Total: ~14-20 hours**

---

## Proposed Changes

### Training Pipeline Component

---

#### [NEW] [hybrid_config.py](file:///c:/Users/Nikhil%20Singhvi/OneDrive/Desktop/Conference1/Smart_Surveillance/Smart-Surveillance-System-for-Vehicle-Crash-Detection/src/training/hybrid_config.py)

Configuration file for hybrid model training parameters:
- Dataset paths and splits
- Hardware-optimized batch sizes
- Learning rate schedules
- Model architecture settings

---

#### [NEW] [dataset_loader.py](file:///c:/Users/Nikhil%20Singhvi/OneDrive/Desktop/Conference1/Smart_Surveillance/Smart-Surveillance-System-for-Vehicle-Crash-Detection/src/training/dataset_loader.py)

Custom PyTorch datasets for training:
- `CrashVideoDataset`: Loads CCD crash/normal videos, extracts frames
- `KITTIDataset`: YOLO-format labels for vehicle detection
- `IDDDataset`: Indian driving dataset loader
- Data augmentation utilities

---

#### [NEW] [vit_classifier.py](file:///c:/Users/Nikhil%20Singhvi/OneDrive/Desktop/Conference1/Smart_Surveillance/Smart-Surveillance-System-for-Vehicle-Crash-Detection/src/training/vit_classifier.py)

Vision Transformer crash classifier:
- MobileViT-S backbone (pre-trained on ImageNet)
- Custom classification head for crash/non-crash
- Feature extraction for fusion layer

---

#### [NEW] [fusion_layer.py](file:///c:/Users/Nikhil%20Singhvi/OneDrive/Desktop/Conference1/Smart_Surveillance/Smart-Surveillance-System-for-Vehicle-Crash-Detection/src/training/fusion_layer.py)

Cross-attention fusion module:
- Combines YOLO detection features with ViT classification features
- Attention-weighted feature aggregation
- Final prediction head

---

#### [NEW] [hybrid_model.py](file:///c:/Users/Nikhil%20Singhvi/OneDrive/Desktop/Conference1/Smart_Surveillance/Smart-Surveillance-System-for-Vehicle-Crash-Detection/src/training/hybrid_model.py)

Complete hybrid model architecture:
- YOLO backbone (YOLOv8n frozen initially)
- ViT encoder (MobileViT-S)
- Fusion layer
- Unified forward pass

---

#### [NEW] [train_hybrid.py](file:///c:/Users/Nikhil%20Singhvi/OneDrive/Desktop/Conference1/Smart_Surveillance/Smart-Surveillance-System-for-Vehicle-Crash-Detection/src/training/train_hybrid.py)

Main training script:
- Phase A: Train YOLO on KITTI + IDD (vehicle detection)
- Phase B: Train ViT on CCD videos (crash classification)
- Phase C: Train fusion layer (both frozen)
- Phase D: End-to-end fine-tuning
- Checkpoint saving and resumption

---

### Inference Service Component

---

#### [NEW] [hybrid_detector.py](file:///c:/Users/Nikhil%20Singhvi/OneDrive/Desktop/Conference1/Smart_Surveillance/Smart-Surveillance-System-for-Vehicle-Crash-Detection/src/services/hybrid_detector.py)

Production inference service:
- Load trained hybrid model
- Sequential processing: YOLO → ViT → Fusion
- Optimized for real-time inference
- Fallback to YOLO-only if ViT fails

---

#### [MODIFY] [enhanced_detection.py](file:///c:/Users/Nikhil%20Singhvi/OneDrive/Desktop/Conference1/Smart_Surveillance/Smart-Surveillance-System-for-Vehicle-Crash-Detection/src/services/enhanced_detection.py)

Integrate hybrid detector:
- Add `use_hybrid_model` config flag
- Replace crash detection with hybrid model when enabled
- Maintain backward compatibility

---

#### [MODIFY] [config.py](file:///c:/Users/Nikhil%20Singhvi/OneDrive/Desktop/Conference1/Smart_Surveillance/Smart-Surveillance-System-for-Vehicle-Crash-Detection/src/config.py)

Add configuration options:
- `USE_HYBRID_MODEL`: Enable/disable hybrid mode
- `HYBRID_MODEL_PATH`: Path to trained weights
- `VIT_CONFIDENCE_THRESHOLD`: ViT classification threshold

---

#### [MODIFY] [task.md](file:///c:/Users/Nikhil%20Singhvi/OneDrive/Desktop/Conference1/Smart_Surveillance/Smart-Surveillance-System-for-Vehicle-Crash-Detection/task.md)

Update Phase 4 task status as implementation progresses.

---

## Implementation Order

```
Step 1: Create training infrastructure
├── hybrid_config.py
├── dataset_loader.py
└── requirements update (timm, torch)

Step 2: Implement model components
├── vit_classifier.py
├── fusion_layer.py
└── hybrid_model.py

Step 3: Training script
└── train_hybrid.py

Step 4: Inference integration
├── hybrid_detector.py
├── enhanced_detection.py (modify)
└── config.py (modify)

Step 5: Verification & testing
└── test_hybrid_model.py
```

---

## Verification Plan

### Automated Tests

1. **Unit Tests for Model Components**
   ```powershell
   cd "c:\Users\Nikhil Singhvi\OneDrive\Desktop\Conference1\Smart_Surveillance\Smart-Surveillance-System-for-Vehicle-Crash-Detection"
   python -m pytest tests/test_hybrid_model.py -v
   ```

2. **Dataset Loading Tests**
   ```powershell
   python -c "from src.training.dataset_loader import CrashVideoDataset; ds = CrashVideoDataset('Datasets/videos-20260206T085121Z-1-002/videos/Crash-1500'); print(f'Loaded {len(ds)} samples')"
   ```

3. **Model Forward Pass Test**
   ```powershell
   python -c "from src.training.hybrid_model import HybridModel; m = HybridModel(); import torch; x = torch.randn(1, 3, 640, 640); print(m(x).shape)"
   ```

### Manual Verification

1. **Training Verification**
   - Run training for 1 epoch on a small subset
   - Verify loss decreases
   - Check checkpoint files are saved to `weights/`

2. **Inference Verification**
   - Enable hybrid mode in config
   - Start the detection service
   - Test with a crash video from CCD dataset
   - Verify detection output includes ViT classification confidence

3. **Browser Test**
   - Open the frontend at http://localhost:3000
   - Start live detection
   - Verify the detection overlay shows "Hybrid" indicator when enabled

---

## Dependencies to Add

```txt
# Add to requirements.txt
timm>=0.9.0          # For MobileViT models
einops>=0.7.0        # For attention operations
```
