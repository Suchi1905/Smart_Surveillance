"""
Configuration management for Smart Surveillance System.

Uses Pydantic Settings for type-safe environment variable loading.
All settings are loaded from .env file or environment variables.
"""

from typing import Optional
from functools import lru_cache
import os
from pathlib import Path


class Settings:
    """
    Application settings loaded from environment variables.
    
    Usage:
        from src.config import get_settings
        settings = get_settings()
        print(settings.bot_token)
    """
    
    def __init__(self):
        # Base paths
        self.base_dir: Path = Path(__file__).resolve().parent.parent
        
        # Telegram Configuration
        self.bot_token: str = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN")
        self.chat_id: str = os.getenv("CHAT_ID", "YOUR_CHAT_ID")
        
        # Model Configuration
        self.model_path: str = os.getenv("MODEL_PATH", "backend/weights/best.pt")
        self.face_model_path: str = os.getenv("FACE_MODEL_PATH", "yolov8n-face.pt")
        
        # API Configuration
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("API_PORT", "8000"))
        self.debug: bool = os.getenv("DEBUG", "true").lower() == "true"
        
        # Database Configuration
        self.database_url: str = os.getenv("DATABASE_URL", "sqlite:///./crash_events.db")
        
        # Detection Settings
        self.confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
        self.severity_buffer_size: int = int(os.getenv("SEVERITY_BUFFER_SIZE", "10"))
        self.severity_iou_threshold: float = float(os.getenv("SEVERITY_IOU_THRESHOLD", "0.3"))
        
        # Privacy Settings
        self.anonymization_enabled: bool = os.getenv("ANONYMIZATION_ENABLED", "true").lower() == "true"
        
        # Alert Settings
        self.alert_cooldown_seconds: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", "10"))
    
    @property
    def telegram_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return (
            self.bot_token != "YOUR_BOT_TOKEN" and 
            self.chat_id != "YOUR_CHAT_ID"
        )
    
    @property
    def model_paths(self) -> list:
        """Get list of possible model paths to search."""
        return [
            os.path.join("backend", "weights", "best.pt"),
            "weights/best.pt",
            "crash_webapp/weights/best.pt",
            os.path.join(self.base_dir, "backend", "weights", "best.pt"),
            os.path.join(self.base_dir, "weights", "best.pt"),
            os.path.join(self.base_dir, "crash_webapp", "weights", "best.pt"),
        ]
    
    def find_model_path(self) -> Optional[str]:
        """Find the first existing model path."""
        for path in self.model_paths:
            if os.path.exists(path):
                return path
        return None


# Load environment variables from .env file
def _load_dotenv():
    """Load .env file if it exists."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


# Initialize on module load
_load_dotenv()


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Settings: Application configuration instance
    """
    return Settings()


# Export settings instance for convenience
settings = get_settings()
