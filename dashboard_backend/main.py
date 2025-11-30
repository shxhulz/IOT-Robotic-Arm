import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import logging
import sys

# Add the current directory to sys.path to ensure imports work
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import init_db, get_db, RobotEvent, Telemetry
from models import EventResponse, TelemetryBase
from websocket_manager import ConnectionManager
from consumer import consume_events
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebSocket Manager
manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    # Start Kafka Consumer in background
    asyncio.create_task(consume_events(manager))
    yield
    # Shutdown (if needed)

app = FastAPI(title="Robotic Arm Dashboard API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return {"status": "online", "service": "dashboard-backend"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, maybe handle client messages if needed
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/events", response_model=List[EventResponse])
def get_events(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    events = db.query(RobotEvent).order_by(RobotEvent.timestamp.desc()).offset(skip).limit(limit).all()
    return events

@app.get("/stats/today")
def get_daily_stats(db: Session = Depends(get_db)):
    # Simple stats for demonstration. 
    # In a real app, use proper SQL queries for aggregation.
    from datetime import datetime, timedelta
    
    last_24h = datetime.utcnow() - timedelta(hours=24)
    
    total_events = db.query(RobotEvent).filter(RobotEvent.timestamp >= last_24h).count()
    
    # Breakdown by object class
    # This is inefficient for large datasets, but fine for MVP
    events = db.query(RobotEvent).filter(RobotEvent.timestamp >= last_24h).all()
    
    class_counts = {}
    for e in events:
        if e.object_class:
            class_counts[e.object_class] = class_counts.get(e.object_class, 0) + 1
            
    return {
        "total_events_24h": total_events,
        "class_breakdown": class_counts
    }

@app.get("/stats/history")
def get_history_stats(db: Session = Depends(get_db)):
    from datetime import datetime, timedelta, timezone
    
    # Get last 24 hours of data (Timezone Aware UTC)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=24)
    
    # Ensure DB timestamps are treated as UTC if they are naive
    # (SQLite usually stores naive strings, so we might need to assume they are UTC)
    events = db.query(RobotEvent).filter(RobotEvent.timestamp >= start_time.replace(tzinfo=None)).all()
    
    # Bucket by hour
    history = {}
    
    # Round start_time down to nearest hour
    current = start_time.replace(minute=0, second=0, microsecond=0)
    
    # Initialize all hours with 0
    while current <= end_time:
        # Use ISO format with timezone info
        iso_key = current.isoformat()
        history[iso_key] = {"time": iso_key, "Plastic": 0, "Metal": 0, "Paper": 0}
        current += timedelta(hours=1)
        
    for e in events:
        # Round event time down to nearest hour
        # Assume event.timestamp is naive UTC, so we add timezone info
        event_time = e.timestamp.replace(tzinfo=timezone.utc)
        event_hour = event_time.replace(minute=0, second=0, microsecond=0)
        iso_key = event_hour.isoformat()
        
        if iso_key in history and e.object_class:
            # Normalize object class name (capitalize)
            cls = e.object_class.capitalize()
            if cls in history[iso_key]:
                history[iso_key][cls] += 1
                
    return list(history.values())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
