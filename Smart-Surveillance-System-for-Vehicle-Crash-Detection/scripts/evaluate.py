"""
Evaluation Metrics Script for Smart Surveillance System.

This script provides comprehensive evaluation metrics for the crash detection model:
- mAP@0.5 and mAP@0.5:0.95
- Precision, Recall, F1-Score
- Severity classification confusion matrix
- Per-class performance breakdown
- Benchmark report generation

Usage:
    python scripts/evaluate.py --weights path/to/weights.pt --data path/to/data.yaml
    python scripts/evaluate.py --weights weights/best.pt --data data.yaml --save-report
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_ap(
    recalls: np.ndarray, 
    precisions: np.ndarray
) -> float:
    """
    Calculate Average Precision using 101-point interpolation.
    
    Args:
        recalls: Array of recall values
        precisions: Array of precision values
        
    Returns:
        Average Precision value
    """
    # Append sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    
    # Compute precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Calculate area under curve using 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max() / 101
    
    return ap


def calculate_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Dict:
    """
    Calculate Precision, Recall, and F1-Score.
    
    Args:
        predictions: List of prediction dicts with 'box', 'confidence', 'class'
        ground_truths: List of ground truth dicts with 'box', 'class'
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with precision, recall, f1, tp, fp, fn counts
    """
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    
    matched_gt = set()
    
    # Sort predictions by confidence (descending)
    sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            if pred.get('class') != gt.get('class'):
                continue
            
            iou = calculate_iou(
                np.array(pred['box']), 
                np.array(gt['box'])
            )
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    # Count unmatched ground truths as false negatives
    fn = len(ground_truths) - len(matched_gt)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }


def calculate_map(
    all_predictions: List[List[Dict]],
    all_ground_truths: List[List[Dict]],
    iou_thresholds: List[float] = None,
    class_names: List[str] = None
) -> Dict:
    """
    Calculate mean Average Precision across all classes.
    
    Args:
        all_predictions: List of predictions per image
        all_ground_truths: List of ground truths per image
        iou_thresholds: List of IoU thresholds (default: [0.5])
        class_names: List of class names
        
    Returns:
        Dictionary with mAP values and per-class APs
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]
    
    if class_names is None:
        # Extract unique classes from ground truths
        all_classes = set()
        for gts in all_ground_truths:
            for gt in gts:
                all_classes.add(gt.get('class', 'unknown'))
        class_names = sorted(list(all_classes))
    
    results = {
        'mAP@0.5': 0.0,
        'mAP@0.5:0.95': 0.0,
        'per_class_ap': {},
        'per_threshold_mAP': {}
    }
    
    # Calculate AP for each IoU threshold
    for iou_thresh in iou_thresholds:
        class_aps = []
        
        for cls in class_names:
            # Collect all predictions and ground truths for this class
            cls_preds = []
            cls_gts = []
            
            for preds, gts in zip(all_predictions, all_ground_truths):
                for pred in preds:
                    if pred.get('class') == cls:
                        cls_preds.append(pred)
                for gt in gts:
                    if gt.get('class') == cls:
                        cls_gts.append(gt)
            
            if len(cls_gts) == 0:
                continue
            
            # Sort predictions by confidence
            cls_preds = sorted(cls_preds, key=lambda x: x['confidence'], reverse=True)
            
            # Calculate precision-recall curve
            precisions = []
            recalls = []
            tp_cumsum = 0
            fp_cumsum = 0
            
            matched = set()
            
            for pred in cls_preds:
                best_iou = 0.0
                best_idx = -1
                
                for idx, gt in enumerate(cls_gts):
                    if idx in matched:
                        continue
                    iou = calculate_iou(
                        np.array(pred['box']), 
                        np.array(gt['box'])
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                
                if best_iou >= iou_thresh:
                    tp_cumsum += 1
                    matched.add(best_idx)
                else:
                    fp_cumsum += 1
                
                precision = tp_cumsum / (tp_cumsum + fp_cumsum)
                recall = tp_cumsum / len(cls_gts)
                
                precisions.append(precision)
                recalls.append(recall)
            
            if len(precisions) > 0:
                ap = calculate_ap(np.array(recalls), np.array(precisions))
                class_aps.append(ap)
                
                if iou_thresh == 0.5:
                    results['per_class_ap'][cls] = ap
        
        if len(class_aps) > 0:
            map_value = np.mean(class_aps)
            results['per_threshold_mAP'][iou_thresh] = map_value
    
    # Calculate mAP@0.5
    if 0.5 in results['per_threshold_mAP']:
        results['mAP@0.5'] = results['per_threshold_mAP'][0.5]
    
    # Calculate mAP@0.5:0.95 (average over IoU thresholds from 0.5 to 0.95, step 0.05)
    iou_range = np.arange(0.5, 1.0, 0.05)
    map_values = [
        results['per_threshold_mAP'].get(t, 0.0) 
        for t in iou_range 
        if t in results['per_threshold_mAP']
    ]
    if len(map_values) > 0:
        results['mAP@0.5:0.95'] = np.mean(map_values)
    
    return results


def calculate_confusion_matrix(
    predictions: List[str],
    ground_truths: List[str],
    classes: List[str]
) -> np.ndarray:
    """
    Calculate confusion matrix for severity classification.
    
    Args:
        predictions: List of predicted class labels
        ground_truths: List of true class labels
        classes: List of class names
        
    Returns:
        Confusion matrix as numpy array
    """
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    for pred, gt in zip(predictions, ground_truths):
        if pred in class_to_idx and gt in class_to_idx:
            pred_idx = class_to_idx[pred]
            gt_idx = class_to_idx[gt]
            matrix[gt_idx, pred_idx] += 1
    
    return matrix


def evaluate_model(
    weights_path: str,
    data_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    device: str = 'cpu'
) -> Dict:
    """
    Evaluate YOLO model on validation dataset.
    
    Args:
        weights_path: Path to model weights
        data_path: Path to data.yaml
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        device: Device to run on
        
    Returns:
        Dictionary with all evaluation metrics
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        return {}
    
    # Load model
    print(f"[📦] Loading model from {weights_path}...")
    model = YOLO(weights_path)
    
    # Run validation
    print(f"[🔍] Running validation on {data_path}...")
    results = model.val(
        data=data_path,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=False
    )
    
    # Extract metrics
    metrics = {
        'model_path': weights_path,
        'data_path': data_path,
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'confidence_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'device': device
        },
        'detection_metrics': {
            'mAP@0.5': float(results.box.map50),
            'mAP@0.5:0.95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
        },
        'per_class_metrics': {},
        'speed': {
            'preprocess_ms': float(results.speed.get('preprocess', 0)),
            'inference_ms': float(results.speed.get('inference', 0)),
            'postprocess_ms': float(results.speed.get('postprocess', 0)),
        }
    }
    
    # Per-class AP
    if hasattr(results.box, 'ap50') and results.names:
        for idx, ap in enumerate(results.box.ap50):
            class_name = results.names.get(idx, f'class_{idx}')
            metrics['per_class_metrics'][class_name] = {
                'AP@0.5': float(ap)
            }
    
    return metrics


def generate_report(metrics: Dict, output_path: str = None) -> str:
    """
    Generate a formatted evaluation report.
    
    Args:
        metrics: Dictionary with evaluation metrics
        output_path: Optional path to save report
        
    Returns:
        Formatted report string
    """
    report_lines = [
        "=" * 60,
        "SMART SURVEILLANCE SYSTEM - EVALUATION REPORT",
        "=" * 60,
        "",
        f"Timestamp: {metrics.get('timestamp', 'N/A')}",
        f"Model: {metrics.get('model_path', 'N/A')}",
        f"Dataset: {metrics.get('data_path', 'N/A')}",
        "",
        "-" * 40,
        "DETECTION PERFORMANCE",
        "-" * 40,
    ]
    
    detection = metrics.get('detection_metrics', {})
    report_lines.extend([
        f"mAP@0.5:       {detection.get('mAP@0.5', 0):.4f}",
        f"mAP@0.5:0.95:  {detection.get('mAP@0.5:0.95', 0):.4f}",
        f"Precision:     {detection.get('precision', 0):.4f}",
        f"Recall:        {detection.get('recall', 0):.4f}",
    ])
    
    # Calculate F1-Score
    p = detection.get('precision', 0)
    r = detection.get('recall', 0)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    report_lines.append(f"F1-Score:      {f1:.4f}")
    
    report_lines.extend([
        "",
        "-" * 40,
        "PER-CLASS PERFORMANCE",
        "-" * 40,
    ])
    
    per_class = metrics.get('per_class_metrics', {})
    if per_class:
        for cls_name, cls_metrics in per_class.items():
            ap = cls_metrics.get('AP@0.5', 0)
            report_lines.append(f"{cls_name:20s} AP@0.5: {ap:.4f}")
    else:
        report_lines.append("No per-class metrics available")
    
    speed = metrics.get('speed', {})
    if any(speed.values()):
        report_lines.extend([
            "",
            "-" * 40,
            "INFERENCE SPEED",
            "-" * 40,
            f"Preprocess:    {speed.get('preprocess_ms', 0):.2f} ms",
            f"Inference:     {speed.get('inference_ms', 0):.2f} ms",
            f"Postprocess:   {speed.get('postprocess_ms', 0):.2f} ms",
            f"Total:         {sum(speed.values()):.2f} ms",
            f"FPS:           {1000 / sum(speed.values()) if sum(speed.values()) > 0 else 0:.1f}",
        ])
    
    report_lines.extend([
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ])
    
    report = "\n".join(report_lines)
    
    # Save report if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"[📄] Report saved to {output_path}")
    
    return report


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description='Evaluate Smart Surveillance crash detection model'
    )
    parser.add_argument(
        '--weights', 
        type=str, 
        default='backend/weights/best.pt',
        help='Path to model weights'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='data.yaml',
        help='Path to data.yaml'
    )
    parser.add_argument(
        '--conf', 
        type=float, 
        default=0.25,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou', 
        type=float, 
        default=0.5,
        help='IoU threshold'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cpu',
        help='Device (cpu or 0 for GPU)'
    )
    parser.add_argument(
        '--save-report', 
        action='store_true',
        help='Save report to file'
    )
    parser.add_argument(
        '--save-json', 
        action='store_true',
        help='Save metrics as JSON'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🚗 Smart Surveillance System - Model Evaluation")
    print("=" * 60 + "\n")
    
    # Check if weights exist
    if not os.path.exists(args.weights):
        print(f"[❌] Error: Weights file not found: {args.weights}")
        print("[ℹ️] Please train a model first using: python modeltrain.py --data data.yaml")
        return
    
    # Evaluate model
    metrics = evaluate_model(
        weights_path=args.weights,
        data_path=args.data,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    
    if not metrics:
        print("[❌] Evaluation failed")
        return
    
    # Generate and print report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"evaluation_report_{timestamp}.txt" if args.save_report else None
    report = generate_report(metrics, report_path)
    print(report)
    
    # Save JSON if requested
    if args.save_json:
        json_path = f"evaluation_metrics_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[📄] Metrics saved to {json_path}")
    
    print("\n[✅] Evaluation complete!")


if __name__ == "__main__":
    main()
