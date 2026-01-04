from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml

class WeatherAugmentationPipeline:
    """
    Synthetic data augmentation pipeline for environmental robustness.
    Creates adversarial training conditions: rain, fog, and low-light scenarios.
    """
    
    def __init__(self):
        # Rain augmentation
        self.rain_transform = A.Compose([
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=3,
                brightness_coefficient=0.9,
                rain_type='drizzle',
                p=1.0
            )
        ])
        
        # Fog augmentation
        self.fog_transform = A.Compose([
            A.RandomFog(
                fog_coef_lower=0.3,
                fog_coef_upper=0.7,
                alpha_coef=0.1,
                p=1.0
            )
        ])
        
        # Low-light enhancement (CLAHE)
        self.lowlight_transform = A.Compose([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, -0.1),
                contrast_limit=0.0,
                p=1.0
            )
        ])
        
        # Combined augmentation (random selection)
        self.combined_transform = A.Compose([
            A.OneOf([
                self.rain_transform,
                self.fog_transform,
            ], p=0.7),
            A.OneOf([
                self.lowlight_transform,
            ], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            )
        ])
    
    def augment_image(self, image_path, output_dir, augmentation_type='combined'):
        """
        Apply augmentation to a single image and save to output directory.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save augmented image
            augmentation_type: 'rain', 'fog', 'lowlight', or 'combined'
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply selected augmentation
        if augmentation_type == 'rain':
            augmented = self.rain_transform(image=image_rgb)['image']
        elif augmentation_type == 'fog':
            augmented = self.fog_transform(image=image_rgb)['image']
        elif augmentation_type == 'lowlight':
            augmented = self.lowlight_transform(image=image_rgb)['image']
        else:  # combined
            augmented = self.combined_transform(image=image_rgb)['image']
        
        # Convert back to BGR for OpenCV
        augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        
        # Save augmented image
        output_path = Path(output_dir) / f"{Path(image_path).stem}_{augmentation_type}{Path(image_path).suffix}"
        cv2.imwrite(str(output_path), augmented_bgr)
        
        return output_path

def augment_dataset(dataset_path, output_path, augmentations_per_image=3):
    """
    Augment entire dataset with weather conditions.
    
    Args:
        dataset_path: Path to original dataset (should contain images/ and labels/ folders)
        output_path: Path to save augmented dataset
        augmentations_per_image: Number of augmentations to apply per image
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'images').mkdir(exist_ok=True)
    (output_path / 'labels').mkdir(exist_ok=True)
    
    pipeline = WeatherAugmentationPipeline()
    
    # Get all images
    image_files = list((dataset_path / 'images').glob('*.jpg')) + \
                  list((dataset_path / 'images').glob('*.png'))
    
    print(f"[📊] Found {len(image_files)} images to augment")
    
    # Copy original images and labels
    print("[📋] Copying original images and labels...")
    for img_path in tqdm(image_files, desc="Copying originals"):
        # Copy image
        shutil.copy(img_path, output_path / 'images' / img_path.name)
        
        # Copy corresponding label if exists
        label_path = dataset_path / 'labels' / (img_path.stem + '.txt')
        if label_path.exists():
            shutil.copy(label_path, output_path / 'labels' / (img_path.stem + '.txt'))
    
    # Apply augmentations
    augmentation_types = ['rain', 'fog', 'lowlight']
    
    print(f"[🔄] Applying {augmentations_per_image} augmentations per image...")
    for img_path in tqdm(image_files, desc="Augmenting"):
        for aug_type in augmentation_types[:augmentations_per_image]:
            try:
                aug_img_path = pipeline.augment_image(
                    img_path,
                    output_path / 'images',
                    augmentation_type=aug_type
                )
                
                if aug_img_path:
                    # Copy corresponding label
                    label_path = dataset_path / 'labels' / (img_path.stem + '.txt')
                    if label_path.exists():
                        aug_label_path = output_path / 'labels' / (aug_img_path.stem + '.txt')
                        shutil.copy(label_path, aug_label_path)
            except Exception as e:
                print(f"[⚠️] Error augmenting {img_path}: {e}")
                continue
    
    print(f"[✅] Augmentation complete! Dataset saved to {output_path}")

def create_augmented_yaml(original_yaml_path, output_yaml_path, augmented_dataset_path):
    """
    Create YAML configuration file for augmented dataset.
    
    Args:
        original_yaml_path: Path to original data.yaml
        output_yaml_path: Path to save new data.yaml
        augmented_dataset_path: Path to augmented dataset
    """
    with open(original_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update paths to point to augmented dataset
    augmented_path = Path(augmented_dataset_path).absolute()
    data['path'] = str(augmented_path)
    data['train'] = str(augmented_path / 'images' / 'train')
    data['val'] = str(augmented_path / 'images' / 'val')
    
    with open(output_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"[✅] Created augmented YAML: {output_yaml_path}")

def train_model(data_yaml="data.yaml", epochs=100, imgsz=640, batch=16, device=0, workers=4):
    """
    Train YOLOv8 model with augmented dataset.
    
    Args:
        data_yaml: Path to data.yaml configuration file
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: Device ID (0 for GPU, 'cpu' for CPU)
        workers: Number of worker threads
    """
    print("[🚀] Starting model training...")
    
    # Load YOLOv8m model
    model = YOLO("yolov8m.pt")
    
    # Train model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        # Additional augmentation settings
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0
    )
    
    print("[✅] Training complete!")
    print(f"[📁] Best weights saved to: {model.trainer.best}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 with weather augmentation')
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to data.yaml')
    parser.add_argument('--dataset', type=str, help='Path to original dataset (for augmentation)')
    parser.add_argument('--augment', action='store_true', help='Apply augmentation before training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=int, default=0, help='Device ID (0 for GPU)')
    
    args = parser.parse_args()
    
    # If augmentation is requested
    if args.augment and args.dataset:
        print("[🔄] Starting dataset augmentation...")
        augmented_path = Path(args.dataset).parent / 'augmented_dataset'
        augment_dataset(args.dataset, augmented_path, augmentations_per_image=3)
        
        # Create augmented YAML
        augmented_yaml = 'data_augmented.yaml'
        create_augmented_yaml(args.data, augmented_yaml, augmented_path)
        
        # Train with augmented dataset
        train_model(
            data_yaml=augmented_yaml,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device
        )
    else:
        # Train with original dataset
        train_model(
            data_yaml=args.data,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device
        )
