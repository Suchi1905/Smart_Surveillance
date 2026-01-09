"""
Database configuration and ORM models for Smart Surveillance System.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database URL from environment or default
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./crash_events.db")

# SQLAlchemy engine
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()


class CrashEvent(Base):
    """
    ORM model for storing crash detection events.
    """
    __tablename__ = "crash_events"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Detection info
    confidence = Column(Float, nullable=False)
    class_name = Column(String(50), nullable=False)
    
    # Severity info
    severity_category = Column(String(20), nullable=False)  # Severe, Moderate, Mild, Monitoring
    severity_index = Column(Float, default=0.0)
    track_id = Column(Integer, nullable=True)
    
    # Bounding box
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    
    # Frame info
    frame_number = Column(Integer)
    
    # Alert status
    alert_sent = Column(Boolean, default=False)
    anonymized = Column(Boolean, default=False)
    
    # Optional metadata
    camera_id = Column(String(50), default="CAM-01")
    location = Column(String(100), default="North Intersection")
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confidence": self.confidence,
            "class_name": self.class_name,
            "severity_category": self.severity_category,
            "severity_index": self.severity_index,
            "track_id": self.track_id,
            "bbox": {
                "x1": self.bbox_x1,
                "y1": self.bbox_y1,
                "x2": self.bbox_x2,
                "y2": self.bbox_y2
            },
            "frame_number": self.frame_number,
            "alert_sent": self.alert_sent,
            "anonymized": self.anonymized,
            "camera_id": self.camera_id,
            "location": self.location
        }


def get_db():
    """
    Dependency for getting database session.
    Yields database session and ensures cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
