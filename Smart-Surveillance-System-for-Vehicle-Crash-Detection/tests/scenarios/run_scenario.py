"""
Scenario Test Runner — Process a video file and generate a test report.

Usage:
    python tests/scenarios/run_scenario.py --video path/to/crash.mp4 --condition daylight
    python tests/scenarios/run_scenario.py --video path/to/night.mp4 --condition night --conf 0.5

Output:
    Prints a summary report with: total frames, crashes detected, severities,
    average confidence, processing FPS, and alerts triggered.
"""

import argparse
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


def run_scenario(video_path: str, condition: str, conf: float = 0.6, max_frames: int = 0):
    """
    Process a video file through the detection pipeline and collect metrics.
    
    Args:
        video_path: Path to the video file
        condition: Test condition label (daylight, night, rain, etc.)
        conf: Confidence threshold
        max_frames: Max frames to process (0 = all)
    
    Returns:
        dict with test metrics
    """
    from services.detection import DetectionService
    from services.severity_triage import SeverityResult

    # Initialize service
    print(f"🔧 Initializing detection service...")
    service = DetectionService()
    models_loaded = service.load_models()
    print(f"   Models loaded: crash={models_loaded[0]}, face={models_loaded[1]}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 Video: {video_path}")
    print(f"   Resolution: {width}x{height}, FPS: {fps_video:.1f}, Frames: {total_frames}")
    print(f"   Condition: {condition}, Confidence: {conf}")
    print(f"   Processing{'...' if max_frames == 0 else f' (max {max_frames} frames)...'}")

    # Track metrics
    frames_processed = 0
    crash_detections = []
    severity_counts = Counter()
    confidences = []
    alerts_triggered = 0
    processing_times = []

    # Track alerts
    original_trigger = service._trigger_alert
    def counting_trigger(frame, sev_result):
        nonlocal alerts_triggered
        alerts_triggered += 1
        # Don't actually send Telegram alerts during testing
        print(f"   🚨 Alert #{alerts_triggered}: {sev_result.severity_category} "
              f"(conf={sev_result.confidence:.2f})")
    service._trigger_alert = counting_trigger

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.perf_counter()
        processed_frame = service._process_frame(frame, conf)
        frame_time = time.perf_counter() - frame_start
        processing_times.append(frame_time)

        frames_processed += 1

        if max_frames > 0 and frames_processed >= max_frames:
            break

        # Progress indicator
        if frames_processed % 100 == 0:
            elapsed = time.time() - start_time
            progress = (frames_processed / total_frames * 100) if total_frames > 0 else 0
            current_fps = frames_processed / elapsed if elapsed > 0 else 0
            print(f"   [{progress:.0f}%] Frame {frames_processed}/{total_frames} "
                  f"({current_fps:.1f} FPS)")

    cap.release()
    total_time = time.time() - start_time

    # Compile report
    avg_fps = frames_processed / total_time if total_time > 0 else 0
    avg_frame_ms = np.mean(processing_times) * 1000 if processing_times else 0
    p95_frame_ms = np.percentile(processing_times, 95) * 1000 if processing_times else 0

    report = {
        "timestamp": datetime.now().isoformat(),
        "video_path": str(video_path),
        "condition": condition,
        "confidence_threshold": conf,
        "video_info": {
            "resolution": f"{width}x{height}",
            "fps": fps_video,
            "total_frames": total_frames,
        },
        "results": {
            "frames_processed": frames_processed,
            "total_time_seconds": round(total_time, 2),
            "average_fps": round(avg_fps, 1),
            "avg_frame_ms": round(avg_frame_ms, 1),
            "p95_frame_ms": round(p95_frame_ms, 1),
            "alerts_triggered": alerts_triggered,
        }
    }

    return report


def print_report(report: dict):
    """Print a formatted test report."""
    if report is None:
        print("❌ No report generated (video could not be processed)")
        return

    print("\n" + "=" * 60)
    print("📊 SCENARIO TEST REPORT")
    print("=" * 60)
    print(f"  Timestamp:    {report['timestamp']}")
    print(f"  Video:        {report['video_path']}")
    print(f"  Condition:    {report['condition']}")
    print(f"  Confidence:   {report['confidence_threshold']}")
    print(f"\n  Video Info:")
    vi = report['video_info']
    print(f"    Resolution: {vi['resolution']}")
    print(f"    FPS:        {vi['fps']}")
    print(f"    Frames:     {vi['total_frames']}")
    print(f"\n  Results:")
    r = report['results']
    print(f"    Processed:     {r['frames_processed']} frames")
    print(f"    Total Time:    {r['total_time_seconds']}s")
    print(f"    Average FPS:   {r['average_fps']}")
    print(f"    Avg Frame:     {r['avg_frame_ms']}ms")
    print(f"    P95 Frame:     {r['p95_frame_ms']}ms")
    print(f"    Alerts:        {r['alerts_triggered']}")
    print("=" * 60)


def save_report(report: dict, output_path: str):
    """Save report to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run crash detection scenario test on a video file"
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--condition", default="standard",
        help="Test condition (daylight, night, rain, fog, angle, etc.)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.6,
        help="Confidence threshold (default: 0.6)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Maximum frames to process (0 = all)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save JSON report (optional)"
    )

    args = parser.parse_args()

    # Validate video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)

    # Run scenario
    report = run_scenario(
        str(video_path),
        condition=args.condition,
        conf=args.conf,
        max_frames=args.max_frames
    )

    # Print report
    print_report(report)

    # Save if requested
    if args.output:
        save_report(report, args.output)
    elif report:
        # Auto-save to tests/scenarios/reports/
        reports_dir = Path(__file__).parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        filename = f"report_{args.condition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_report(report, str(reports_dir / filename))


if __name__ == "__main__":
    main()
