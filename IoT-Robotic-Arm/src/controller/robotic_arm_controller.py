import time

import cv2

from src.config.config import (
    CAMER_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CENTER_THRESHOLD,
    DETECTION_CONFIDENCE_THRESHOLD,
    SERVO_PORT,
)
from src.controller.servo_controller import ServoControl
from src.detector.object_detector import ObjectDetector
from src.utils.logger import get_logger

import json
from datetime import datetime
from kafka import KafkaProducer

logger = get_logger(__name__)

class SafeProducer:
    def __init__(self, bootstrap_servers='localhost:9092'):
        self.producer = None
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                request_timeout_ms=1000,
                api_version_auto_timeout_ms=1000
            )
            logger.info("Kafka Producer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Kafka Producer: {e}. Running in standalone mode.")

    def send(self, topic, data):
        if self.producer:
            try:
                # Add timestamp if not present
                if 'timestamp' not in data:
                    data['timestamp'] = datetime.utcnow().isoformat()
                self.producer.send(topic, data)
            except Exception as e:
                logger.warning(f"Failed to send Kafka message: {e}")



class RoboticArmController:
    def __init__(self):
        """Initialize the robotic arm controller with camera, object detector, and servo control."""
        logger.info("Initializing Robotic Arm Controller...")
        logger.debug(
            f"Setting up camera at {CAMERA_INDEX} with resolution {CAMER_WIDTH}x{CAMERA_HEIGHT}"
        )
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        logger.debug("Camera initialized successfully.")

        logger.debug("Initializing Object Detector...")
        self.object_detector = ObjectDetector(conf=DETECTION_CONFIDENCE_THRESHOLD)
        logger.debug("Object Detector initialized successfully.")

        logger.debug(f"Initializing Servo Control at port {SERVO_PORT} ...")
        try:
            self.servo_control = ServoControl(COM=SERVO_PORT)
            logger.info("Servo Control initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Servo Control: {e}")
            if "PermissionError" in str(e) or "Access is denied" in str(e):
                logger.error(f"PORT {SERVO_PORT} is likely in use by another process. Please close any other applications using this port.")
            self.servo_control = None
        
        self.producer = SafeProducer()
        self.robot_id = "robot_1"
        self.frame_width = CAMER_WIDTH
        self.frame_height = CAMERA_HEIGHT
        self.center_threshold = CENTER_THRESHOLD

    def get_distance(self) -> str | None:
        """Get distance from ultrasonic sensor."""
        if not self.servo_control:
            return None
            
        try:
            self.servo_control.ser.write("DIST\n".encode("utf-8"))
            time.sleep(0.1)

            while self.servo_control.ser.in_waiting > 0:
                response = self.servo_control.ser.readline().decode("utf-8").strip()
                if response.startswith("DISTC"):
                    distance = int(response[5:])
                    logger.info(f"Distance measured: {distance}cm")
                    self.producer.send("robot_telemetry", {
                        "robot_id": self.robot_id,
                        "distance": distance
                    })
                    return distance
            return None
        except Exception as e:
            logger.error(f"Error getting distance: {e}")
            return None

    def center_object(self, bbox):
        """
        Moves the base servo to center the object in the frame.
        args:
            bbox (BoundingBox): The bounding box of the detected object.
        """
        if not bbox:
            return False

        center_x, center_y = bbox.center
        frame_center_x = self.frame_width / 2

        distance_from_center = center_x - frame_center_x

        if abs(distance_from_center) <= self.center_threshold:
            logger.info("Object centered")
            return True

        if distance_from_center > 0:
            logger.info("Moving base servo right")
            if self.servo_control:
                self.servo_control.baseServoRight()
        else:
            logger.info("Moving base servo left")
            if self.servo_control:
                self.servo_control.baseServoLeft()

        return False

    def execute_pickup_sequence(self, distance, object_class):
        """
        Execute the pickup sequence based on distance and object class.
        args:
            distance (int): Distance to the object in cm.
        """
        if not self.servo_control:
            logger.warning("Servo control not initialized, skipping pickup sequence")
            return False

        try:
            distance_str = str(int(distance))

            if distance_str in self.servo_control.positionData:
                logger.info(f"Moving to position for distance {distance}cm")
                self.servo_control.setCircle(distance_str)
            else:
                logger.warning(
                    f"No predefined position for distance {distance}cm, finding nearest position"
                )
                nearest_position = self.find_nearest_position(distance)
                if nearest_position:
                    logger.info(
                        f"Using nearest position {nearest_position} for distance {distance}cm"
                    )
                    self.servo_control.setCircle(nearest_position)
                else:
                    logger.warning("No nearby positions found, using default position")
                    if hasattr(self.servo_control, "setDefaultPosition"):
                        self.servo_control.setDefaultPosition()
                    else:
                        known_positions = list(self.servo_control.positionData.keys())
                        if known_positions:
                            default_pos = known_positions[0]
                            logger.info(f"Using first known position: {default_pos}")
                            logger.info("Opening gripper")
                            self.servo_control.openGripper()
                            self.servo_control.setCircle(default_pos)
                        else:
                            logger.error("No positions available in positionData")
                            return False

            logger.info(f"Closing gripper to grab object of class {object_class}")
            self.servo_control.closeGripper()

            success = self.execute_disposal_sequence(object_class)

            logger.info("Returning to rest position")
            if hasattr(self.servo_control, "setRestPosition"):
                self.servo_control.setRestPosition()
            else:
                logger.info("Returning to neutral position")
                self.servo_control.setNeutralPosition()

            return success
        except Exception as e:
            logger.error(f"Error in pickup sequence: {e}")
            return False

    def find_nearest_position(self, distance):
        """Find the nearest available position for a given distance."""
        try:
            distance = int(distance)
            available_positions = self.servo_control.positionData.keys()

            positions_int = {}
            for pos in available_positions:
                try:
                    pos_int = int(pos)
                    positions_int[pos] = pos_int
                except ValueError:
                    continue

            if not positions_int:
                return None

            closest_pos = min(
                positions_int.keys(), key=lambda x: abs(positions_int[x] - distance)
            )

            if abs(positions_int[closest_pos] - distance) <= 30:
                return closest_pos
            else:
                return None
        except Exception as e:
            logger.error(f"Error finding nearest position: {e}")
            return None

    def execute_disposal_sequence(self, object_class):
        """Execute the disposal sequence based on object class."""
        try:
            logger.info(f"Executing disposal sequence for object class: {object_class}")

            disposal_positions = {
                "paper": "paperDisposal",
                "metal": "metalDisposal",
                "plastic": "plasticDisposal",
            }

            if object_class in disposal_positions:
                disposal_position = disposal_positions[object_class]
                logger.info(
                    f"Using disposal position '{disposal_position}' for object class '{object_class}'"
                )

                if disposal_position in self.servo_control.positionData:
                    logger.info(f"Moving to disposal position: {disposal_position}")
                    self.servo_control.setCircle(disposal_position)
                    time.sleep(1.5)

                    logger.info("Opening gripper to release object")
                    self.servo_control.openGripper()
                    time.sleep(1)

                    return True
                else:
                    logger.warning(
                        f"Disposal position '{disposal_position}' not found in positionData"
                    )
            else:
                logger.warning(
                    f"No disposal position defined for object class '{object_class}', using default"
                )

                if "paperDisposal" in self.servo_control.positionData:
                    logger.info("Using paperDisposal as default disposal position")
                    self.servo_control.setCircle("paperDisposal")
                else:
                    logger.info("Using manual disposal position")
                    self.servo_control.moveServoSingle(1, 90)
                    time.sleep(0.5)
                    self.servo_control.moveServoSingle(2, 120)
                    time.sleep(0.5)
                    self.servo_control.moveServoSingle(3, 40)

                time.sleep(1.5)

                logger.info("Opening gripper to release object")
                self.servo_control.openGripper()
                time.sleep(1)

                return True

        except Exception as e:
            logger.error(f"Error in disposal sequence: {e}")
            try:
                self.servo_control.openGripper()
            except Exception as e2:
                logger.error(f"Failed to open gripper: {e2}")
            return False

    def get_disposal_position(self, object_class):
        """Get the disposal position for a specific object class."""
        logger.info(f"Getting disposal position for object class: {object_class}")
        disposal_positions = {
            "paper": "paperDisposal",
            "metal": "metalDisposal",
            "plastic": "plasticDisposal",
        }

        if object_class in disposal_positions:
            position_key = disposal_positions[object_class]
            if position_key in self.servo_control.positionData:
                logger.info(
                    f"Found disposal position '{position_key}' for object class '{object_class}'"
                )
                return position_key

        logger.warning(
            f"Using default disposal position for object class '{object_class}'"
        )
        return "paperDisposal"

    def run(self):
        """Main control loop for the robotic arm."""
        logger.info("Starting robotic arm control loop")

        try:
            executing_sequence = False
            consecutive_camera_errors = 0

            while True:
                if not executing_sequence:
                    ret, frame = self.camera.read()
                    if not ret:
                        consecutive_camera_errors += 1
                        logger.error(
                            f"Failed to read frame from camera (attempt {consecutive_camera_errors})"
                        )

                        if consecutive_camera_errors >= 5:
                            logger.warning(
                                "Multiple camera read failures, attempting to reconnect camera"
                            )
                            self.reconnect_camera()
                            consecutive_camera_errors = 0

                        time.sleep(0.5)
                        continue
                    else:
                        consecutive_camera_errors = 0

                    annotated_frame, detections = self.object_detector.detect(frame)

                    cv2.imshow("Object Detection", annotated_frame)

                    if detections:
                        detections.sort(key=lambda x: x.confidence, reverse=True)
                        target_bbox = detections[0]
                        object_class = target_bbox.class_name
                        
                        self.producer.send("robot_events", {
                            "robot_id": self.robot_id,
                            "event_type": "detection",
                            "object_class": object_class,
                            "details": f"Detected {object_class}"
                        })

                        if self.center_object(target_bbox):
                            distance = self.get_distance()
                            if distance is not None:
                                executing_sequence = True
                                logger.info("Starting pickup and disposal sequence")
                                self.producer.send("robot_events", {
                                    "robot_id": self.robot_id,
                                    "event_type": "pickup_start",
                                    "object_class": object_class,
                                    "details": "Starting pickup sequence"
                                })

                                try:
                                    if self.execute_pickup_sequence(
                                        distance + 1, object_class
                                    ):
                                        logger.info(
                                            "Pickup and disposal sequence completed successfully"
                                        )
                                        self.producer.send("robot_events", {
                                            "robot_id": self.robot_id,
                                            "event_type": "pickup_success",
                                            "object_class": object_class,
                                            "details": "Pickup successful"
                                        })
                                    else:
                                        logger.warning(
                                            "Pickup and disposal sequence failed"
                                        )
                                        self.producer.send("robot_events", {
                                            "robot_id": self.robot_id,
                                            "event_type": "pickup_fail",
                                            "object_class": object_class,
                                            "details": "Pickup failed"
                                        })
                                finally:
                                    executing_sequence = False
                                    logger.info(
                                        "Sequence completed, resuming object detection"
                                    )
                                    self.reconnect_camera()
                else:
                    time.sleep(0.1)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            logger.info("Control loop interrupted by user")
        except Exception as e:
            logger.error(f"Error in control loop: {e}")
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            self.servo_control.close()
            logger.info("Robotic arm control stopped")

    def reconnect_camera(self):
        """Reconnect to the camera if connection is lost."""
        try:
            if self.camera is not None:
                self.camera.release()

            logger.info("Reconnecting to camera...")
            self.camera = cv2.VideoCapture(CAMERA_INDEX)

            if self.camera.isOpened():
                self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(
                    f"Camera reconnected successfully with resolution: {self.frame_width}x{self.frame_height}"
                )
                time.sleep(1)
                return True
            else:
                logger.error("Failed to reconnect to camera")
                return False
        except Exception as e:
            logger.error(f"Error reconnecting to camera: {e}")
            return False
