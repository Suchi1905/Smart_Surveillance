"""
Video streaming router for FastAPI.

Provides MJPEG video stream with crash detection overlay.
"""

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

router = APIRouter(tags=["Video"])


# Global reference to frame generator (set by main.py)
_frame_generator = None


def set_frame_generator(generator_func):
    """Set the frame generator function."""
    global _frame_generator
    _frame_generator = generator_func


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
