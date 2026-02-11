"""
Video streaming router for FastAPI.

Provides MJPEG video stream with crash detection overlay.
Supports webcam, URL, and local file upload sources.
"""

import os
import uuid
import logging
from pathlib import Path
from fastapi import APIRouter, Query, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

router = APIRouter(tags=["Video"])
logger = logging.getLogger(__name__)

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "video_cache" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

# Global reference to frame generator (set by main.py)
_frame_generator = None
_detection_service = None

def set_frame_generator(generator_func):
    """Set the frame generator function."""
    global _frame_generator
    _frame_generator = generator_func

def set_detection_service(service):
    """Set the detection service instance."""
    global _detection_service
    _detection_service = service


@router.get("/video")
async def video_stream(
    conf: float = Query(0.6, ge=0.1, le=1.0, description="Confidence threshold")
):
    """
    MJPEG video stream with real-time crash detection.
    
    - **conf**: Confidence threshold for detections (0.1 to 1.0)
    
    Returns a multipart MJPEG stream that can be viewed in a browser
    or embedded in an img tag.
    """
    if _frame_generator is None:
        async def error_generator():
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nVideo stream not initialized\r\n'
        
        return StreamingResponse(
            error_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    return StreamingResponse(
        _frame_generator(conf),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.post("/video/stop")
async def stop_video_stream():
    """
    Stop the video stream and release camera resources.
    """
    global _detection_service
    
    if _detection_service and hasattr(_detection_service, 'stop_stream'):
        try:
            _detection_service.stop_stream()
            return {"status": "stopped", "message": "Camera release signal sent"}
        except Exception as e:
            return {"status": "error", "message": f"Error stopping stream: {str(e)}"}
            
    return {"status": "error", "message": "Detection service not initialized"}


@router.post("/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a local video file for crash detection analysis.
    
    Accepts .mp4, .avi, .mkv, .mov, .wmv, .flv, .webm files up to 500MB.
    Returns the filename to use with /video/file endpoint.
    """
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate unique filename
    unique_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = UPLOAD_DIR / unique_name
    
    # Save file
    try:
        total_size = 0
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    f.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="File too large (max 500MB)")
                f.write(chunk)
        
        logger.info(f"Video uploaded: {unique_name} ({total_size / (1024*1024):.1f} MB)")
        
        return JSONResponse({
            "status": "uploaded",
            "filename": unique_name,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "path": str(file_path)
        })
    except HTTPException:
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/video/file")
async def video_stream_from_file(
    filename: str = Query(..., description="Uploaded video filename"),
    conf: float = Query(0.6, ge=0.1, le=1.0, description="Confidence threshold")
):
    """
    MJPEG video stream with real-time crash detection from an uploaded file.
    
    Use /video/upload first to upload a file, then pass the returned filename here.
    """
    if _detection_service is None:
        async def error_generator():
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nDetection service not initialized\r\n'
        return StreamingResponse(
            error_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {filename}")
    
    if not hasattr(_detection_service, 'generate_frames_from_url'):
        async def error_generator():
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nFile streaming not supported\r\n'
        return StreamingResponse(
            error_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    # generate_frames_from_url works with local file paths too (OpenCV accepts file paths)
    return StreamingResponse(
        _detection_service.generate_frames_from_url(str(file_path), conf),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/video/uploads")
async def list_uploaded_videos():
    """List all uploaded video files available for analysis."""
    files = []
    for f in UPLOAD_DIR.iterdir():
        if f.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append({
                "filename": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2)
            })
    return {"files": files}


@router.get("/video/url")
async def video_stream_from_url(
    source: str = Query(..., description="Video source URL (YouTube, RTSP, or direct video URL)"),
    conf: float = Query(0.6, ge=0.1, le=1.0, description="Confidence threshold")
):
    """
    MJPEG video stream with real-time crash detection from URL source.
    
    Supports:
    - **YouTube URLs**: Auto-extracts stream URL via yt-dlp
    - **RTSP streams**: Direct RTSP URLs (rtsp://...)
    - **Video files**: Direct URLs to .mp4, .avi, etc.
    """
    if _detection_service is None:
        async def error_generator():
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nDetection service not initialized\r\n'
        
        return StreamingResponse(
            error_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    if not hasattr(_detection_service, 'generate_frames_from_url'):
        async def error_generator():
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nURL streaming not supported\r\n'
        
        return StreamingResponse(
            error_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    return StreamingResponse(
        _detection_service.generate_frames_from_url(source, conf),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

