from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./dashboard.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class RobotEvent(Base):
    __tablename__ = "robot_events"

    id = Column(Integer, primary_key=True, index=True)
    robot_id = Column(String, index=True)
    event_type = Column(String, index=True)  # detection, pickup_start, pickup_success, pickup_fail
    object_class = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    details = Column(String, nullable=True)

class Telemetry(Base):
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, index=True)
    robot_id = Column(String, index=True)
    distance = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
