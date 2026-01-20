"""
Video streaming router for FastAPI.

Provides MJPEG video stream with crash detection overlay.
"""

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

router = APIRouter(tags=["Video"])


# Global reference to frame generator (set by main.py)
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
        # Return error frame if generator not set
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
    
    Example:
    - `/video/url?source=https://www.youtube.com/watch?v=VIDEO_ID&conf=0.6`
    - `/video/url?source=rtsp://camera-ip/stream&conf=0.5`
    
    - **source**: Video source URL
    - **conf**: Confidence threshold for detections (0.1 to 1.0)
    
    Returns a multipart MJPEG stream that can be viewed in a browser
    or embedded in an img tag.
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
