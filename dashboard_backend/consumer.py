import asyncio
import json
import logging
from aiokafka import AIOKafkaConsumer
from sqlalchemy.orm import Session
from database import SessionLocal, RobotEvent, Telemetry
from datetime import datetime

logger = logging.getLogger(__name__)

KAFKA_TOPIC_EVENTS = "robot_events"
KAFKA_TOPIC_TELEMETRY = "robot_telemetry"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

async def consume_events(websocket_manager):
    consumer = AIOKafkaConsumer(
        KAFKA_TOPIC_EVENTS,
        KAFKA_TOPIC_TELEMETRY,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="dashboard_group",
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    try:
        await consumer.start()
        logger.info("Kafka Consumer started")
    except Exception as e:
        logger.error(f"Failed to start Kafka Consumer: {e}")
        return

    try:
        async for msg in consumer:
            data = msg.value
            topic = msg.topic
            
            # Broadcast to WebSockets
            await websocket_manager.broadcast(json.dumps({
                "topic": topic,
                "data": data
            }))

            # Save to DB
            db: Session = SessionLocal()
            try:
                if topic == KAFKA_TOPIC_EVENTS:
                    event = RobotEvent(
                        robot_id=data.get("robot_id", "unknown"),
                        event_type=data.get("event_type"),
                        object_class=data.get("object_class"),
                        details=data.get("details"),
                        timestamp=datetime.fromisoformat(data.get("timestamp")) if data.get("timestamp") else datetime.utcnow()
                    )
                    db.add(event)
                elif topic == KAFKA_TOPIC_TELEMETRY:
                    # Optional: Don't save every single telemetry point to avoid DB bloat, 
                    # or save it if historical telemetry is needed. 
                    # For now, let's save it.
                    telemetry = Telemetry(
                        robot_id=data.get("robot_id", "unknown"),
                        distance=data.get("distance"),
                        timestamp=datetime.fromisoformat(data.get("timestamp")) if data.get("timestamp") else datetime.utcnow()
                    )
                    db.add(telemetry)
                
                db.commit()
            except Exception as e:
                logger.error(f"Error saving to DB: {e}")
            finally:
                db.close()

    except Exception as e:
        logger.error(f"Kafka Consumer error: {e}")
    finally:
        await consumer.stop()
