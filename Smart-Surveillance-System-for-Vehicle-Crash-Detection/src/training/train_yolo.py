"""
Phase 4A: YOLO Backbone Training Script

Trains YOLOv8 on KITTI + IDD datasets for vehicle detection.
Optimized for RTX 4050 (6GB VRAM).

Usage:
    python train_yolo.py --epochs 50 --batch 4
    python train_yolo.py --prepare-data  # Just prepare dataset
"""

import os
import sys
import argparse
import shutil
import logging
from pathlib import Path
from typing import Optional
import random

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLODatasetPreparer:
    """
    Prepares KITTI dataset in YOLO format for training.
    """
    
    # KITTI classes to use (vehicle-focused)
    CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist']
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.dataset.kitti_labels_dir).parent / "yolo_format"
    
    def prepare_kitti(self) -> Path:
        """Convert KITTI to YOLO format and create data.yaml"""
        logger.info("Preparing KITTI dataset in YOLO format...")
        
        # Create output directory structure
        train_images = self.output_dir / "images" / "train"
        train_labels = self.output_dir / "labels" / "train"
        val_images = self.output_dir / "images" / "val"
        val_labels = self.output_dir / "labels" / "val"
        
        for d in [train_images, train_labels, val_images, val_labels]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        kitti_images = self.config.dataset.kitti_images_dir
        kitti_labels = self.config.dataset.kitti_labels_dir
        
        # Check for nested structure (KITTI sometimes has training/image_2)
        if (kitti_images / "training" / "image_2").exists():
            image_dir = kitti_images / "training" / "image_2"
        elif (kitti_images / "image_2").exists():
            image_dir = kitti_images / "image_2"
        else:
            image_dir = kitti_images
            
        if (kitti_labels / "training" / "label_2").exists():
            label_dir = kitti_labels / "training" / "label_2"
        elif (kitti_labels / "label_2").exists():
            label_dir = kitti_labels / "label_2"
        else:
            label_dir = kitti_labels
        
        # Get all images
        image_files = sorted(list(image_dir.glob("*.png")))
        if not image_files:
            image_files = sorted(list(image_dir.glob("**/*.png")))
        
        logger.info(f"Found {len(image_files)} images in {image_dir}")
        
        # Split into train/val
        random.shuffle(image_files)
        split_idx = int(len(image_files) * 0.9)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Process training set
        logger.info(f"Processing {len(train_files)} training images...")
        for img_path in train_files:
            self._process_image(img_path, label_dir, train_images, train_labels)
        
        # Process validation set
        logger.info(f"Processing {len(val_files)} validation images...")
        for img_path in val_files:
            self._process_image(img_path, label_dir, val_images, val_labels)
        
        # Create data.yaml
        data_yaml = self.output_dir / "data.yaml"
        yaml_content = f"""
# KITTI Dataset in YOLO format
path: {self.output_dir.absolute()}
train: images/train
val: images/val

nc: {len(self.CLASSES)}
names: {self.CLASSES}
"""
        data_yaml.write_text(yaml_content.strip())
        logger.info(f"Created data.yaml at {data_yaml}")
        
        return data_yaml
    
    def _process_image(
        self, 
        img_path: Path, 
        label_dir: Path, 
        out_images: Path, 
        out_labels: Path
    ):
        """Process single image and its label"""
        import cv2
        
        # Copy image
        dst_img = out_images / img_path.name
        if not dst_img.exists():
            shutil.copy(img_path, dst_img)
        
        # Find corresponding label
        label_name = img_path.stem + ".txt"
        src_label = label_dir / label_name
        dst_label = out_labels / label_name
        
        if not src_label.exists():
            # Create empty label file
            dst_label.write_text("")
            return
        
        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            return
        h, w = img.shape[:2]
        
        # Convert KITTI label to YOLO format
        yolo_labels = []
        with open(src_label, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                
                class_name = parts[0]
                if class_name not in self.CLASSES:
                    continue
                
                class_id = self.CLASSES.index(class_name)
                
                # KITTI bbox: left, top, right, bottom
                left = float(parts[4])
                top = float(parts[5])
                right = float(parts[6])
                bottom = float(parts[7])
                
                # Convert to YOLO format (normalized)
                x_center = ((left + right) / 2) / w
                y_center = ((top + bottom) / 2) / h
                width = (right - left) / w
                height = (bottom - top) / h
                
                # Clamp values
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Write YOLO label file
        dst_label.write_text("\n".join(yolo_labels))


def train_yolo(
    data_yaml: Path,
    epochs: int = 50,
    batch_size: int = 4,
    image_size: int = 640,
    device: str = "0",
    project: str = "weights/yolo_training",
    name: str = "phase4_yolo",
    resume: bool = False
):
    """
    Train YOLOv8 model on prepared dataset.
    
    Args:
        data_yaml: Path to data.yaml
        epochs: Number of training epochs
        batch_size: Batch size (keep low for RTX 4050)
        image_size: Input image size
        device: GPU device
        project: Output project directory
        name: Run name
        resume: Resume from last checkpoint
    """
    logger.info("="*60)
    logger.info("PHASE 4A: YOLO BACKBONE TRAINING")
    logger.info("="*60)
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Image size: {image_size}")
    logger.info("="*60)
    
    # Load YOLOv8 nano (optimized for low VRAM)
    model = YOLO("yolov8n.pt")
    
    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        resume=resume,
        
        # Memory optimization for RTX 4050
        amp=True,  # Mixed precision
        cache=False,  # Don't cache images (saves RAM)
        workers=0,  # Avoid Windows multiprocessing CUDA issues
        
        # Augmentation
        augment=True,
        mosaic=0.5,
        mixup=0.0,
        
        # Learning rate
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        
        # Logging
        verbose=True,
        plots=True
    )
    
    logger.info("Training complete!")
    logger.info(f"Best weights saved to: {project}/{name}/weights/best.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="YOLO Training for Phase 4A")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare KITTI dataset only")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="GPU device")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    args = parser.parse_args()
    
    # Import config
    from hybrid_config import config
    
    # Prepare dataset
    preparer = YOLODatasetPreparer(config)
    data_yaml = preparer.output_dir / "data.yaml"
    
    if not data_yaml.exists() or args.prepare_data:
        data_yaml = preparer.prepare_kitti()
        if args.prepare_data:
            logger.info("Dataset preparation complete. Exiting.")
            return
    
    # Train model
    train_yolo(
        data_yaml=data_yaml,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        device=args.device,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
