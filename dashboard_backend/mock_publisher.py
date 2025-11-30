import time
import json
import random
import datetime
from kafka import KafkaProducer

KAFKA_TOPIC_EVENTS = "robot_events"
KAFKA_TOPIC_TELEMETRY = "robot_telemetry"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

def json_serializer(data):
    return json.dumps(data).encode("utf-8")

def run_mock_producer():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=json_serializer
    )

    print("Starting Mock Producer...")
    
    object_classes = ["paper", "metal", "plastic"]
    event_types = ["detection", "pickup_start", "pickup_success", "pickup_fail"]

    try:
        while True:
            # Simulate Telemetry (frequent)
            telemetry_data = {
                "robot_id": "robot_1",
                "distance": round(random.uniform(10.0, 50.0), 2),
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
            producer.send(KAFKA_TOPIC_TELEMETRY, telemetry_data)
            print(f"Sent Telemetry: {telemetry_data['distance']}cm")

            # Simulate Events (random)
            if random.random() < 0.3:  # 30% chance of event
                event_type = random.choice(event_types)
                obj_class = random.choice(object_classes) if event_type != "pickup_fail" else None
                
                event_data = {
                    "robot_id": "robot_1",
                    "event_type": event_type,
                    "object_class": obj_class,
                    "details": f"Mock event {event_type}",
                    "timestamp": datetime.datetime.utcnow().isoformat()
                }
                producer.send(KAFKA_TOPIC_EVENTS, event_data)
                print(f"Sent Event: {event_type} - {obj_class}")

            time.sleep(1)  # 1 second interval

    except KeyboardInterrupt:
        print("Stopping Mock Producer...")
    finally:
        producer.close()

if __name__ == "__main__":
    run_mock_producer()
