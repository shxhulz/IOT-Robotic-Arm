from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class EventBase(BaseModel):
    robot_id: str
    event_type: str
    object_class: Optional[str] = None
    details: Optional[str] = None
    timestamp: Optional[datetime] = None

class EventCreate(EventBase):
    pass

class EventResponse(EventBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True

class TelemetryBase(BaseModel):
    robot_id: str
    distance: float
    timestamp: Optional[datetime] = None

class TelemetryCreate(TelemetryBase):
    pass
