"""
Multi-threaded robotic arm controller.

Threads:
  1. camera_thread  – Captures RTSP frames → frame_queue
  2. vision_thread  – Detects objects → cmd_queue  (when stable & IDLE)
  3. robot_thread   – Sends commands over serial → waits for OK

Shared state (robot_state, target_locked, detection_counter) is protected
by a threading.Lock.
"""

import threading
import time
from queue import Empty, Full, Queue
from io import BytesIO

import cv2
from minio import Minio

from config.config import (
    CAMERA_HEIGHT,
    CAMERA_STREAM_URL,
    CAMERA_WIDTH,
    CENTER_THRESHOLD,
    CMD_QUEUE_SIZE,
    DETECTION_CONFIDENCE_THRESHOLD,
    DETECTION_STABILITY_THRESHOLD,
    FRAME_QUEUE_SIZE,
    SERVO_PORT,
    WINDOW_NAME,
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_BUCKET_NAME,
    MINIO_UPLOAD_INTERVAL,
    MINIO_SECURE,
)
from controller.servo_controller import ServoControl
from detector.object_detector import ObjectDetector
from utils.logger import get_logger

import json
from datetime import datetime
from kafka import KafkaProducer

logger = get_logger(__name__)


# ── Kafka helper ──────────────────────────────────────────────────────────


class SafeProducer:
    def __init__(self, bootstrap_servers="localhost:9092"):
        self.producer = None
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=1000,
                api_version_auto_timeout_ms=1000,
            )
            logger.info("Kafka Producer initialized successfully")
        except Exception as e:
            logger.warning(
                f"Failed to initialize Kafka Producer: {e}. Running in standalone mode."
            )

    def send(self, topic, data):
        if self.producer:
            try:
                if "timestamp" not in data:
                    data["timestamp"] = datetime.utcnow().isoformat()
                self.producer.send(topic, data)
            except Exception as e:
                logger.warning(f"Failed to send Kafka message: {e}")


# ── Main controller class ────────────────────────────────────────────────


class RoboticArmController:
    """
    Orchestrates three threads: camera capture, vision processing,
    and robot serial control via thread-safe queues.
    """

    def __init__(self):
        logger.info("Initializing Robotic Arm Controller...")

        # ── Queues ──
        self.frame_queue: Queue = Queue(maxsize=FRAME_QUEUE_SIZE)
        self.cmd_queue: Queue = Queue(maxsize=CMD_QUEUE_SIZE)
        logger.debug(
            f"Queues created – frame_queue(maxsize={FRAME_QUEUE_SIZE}), "
            f"cmd_queue(maxsize={CMD_QUEUE_SIZE})"
        )

        # ── Shared state (guarded by _state_lock) ──
        self._state_lock = threading.Lock()
        self._robot_state = "IDLE"  # "IDLE" | "MOVING"
        self._target_locked = False
        self._detection_counter = 0

        # ── Latest annotated frame for display (guarded by _frame_lock) ──
        self._frame_lock = threading.Lock()
        self._display_frame = None

        # ── FPS tracking for camera overlay ──
        self._fps = 0.0

        # ── Shutdown event ──
        self._shutdown = threading.Event()

        # ── Object detector ──
        logger.debug("Initializing Object Detector...")
        self.object_detector = ObjectDetector(conf=DETECTION_CONFIDENCE_THRESHOLD)
        logger.debug("Object Detector initialized successfully.")

        # ── Servo control ──
        logger.debug(f"Initializing Servo Control at port {SERVO_PORT} ...")
        try:
            self.servo_control = ServoControl(COM=SERVO_PORT)
            logger.info("Servo Control initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Servo Control: {e}")
            if "PermissionError" in str(e) or "Access is denied" in str(e):
                logger.error(
                    f"PORT {SERVO_PORT} is likely in use by another process. "
                    "Please close any other applications using this port."
                )
            self.servo_control = None

        # ── Kafka producer ──
        self.producer = SafeProducer()
        self.robot_id = "robot_1"

        # ── Minio Client ──
        try:
            self.minio_client = Minio(
                MINIO_ENDPOINT,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=MINIO_SECURE,
            )
            # Check if bucket exists, create if not
            if not self.minio_client.bucket_exists(MINIO_BUCKET_NAME):
                self.minio_client.make_bucket(MINIO_BUCKET_NAME)
            logger.info(f"Minio client initialized. Bucket: {MINIO_BUCKET_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize Minio client: {e}")
            self.minio_client = None

        # ── Frame dimensions ──
        self.frame_width = CAMERA_WIDTH
        self.frame_height = CAMERA_HEIGHT
        self.center_threshold = CENTER_THRESHOLD

    # ── State helpers (thread-safe) ───────────────────────────────────────

    @property
    def robot_state(self):
        with self._state_lock:
            return self._robot_state

    @robot_state.setter
    def robot_state(self, value):
        with self._state_lock:
            old = self._robot_state
            self._robot_state = value
        if old != value:
            logger.info(f"robot_state: {old} → {value}")

    @property
    def target_locked(self):
        with self._state_lock:
            return self._target_locked

    @target_locked.setter
    def target_locked(self, value):
        with self._state_lock:
            self._target_locked = value
        logger.debug(f"target_locked = {value}")

    @property
    def detection_counter(self):
        with self._state_lock:
            return self._detection_counter

    @detection_counter.setter
    def detection_counter(self, value):
        with self._state_lock:
            self._detection_counter = value

    # ── Thread 1: Camera capture ──────────────────────────────────────────

    def _upload_frame_to_minio(self, frame):
        """Helper to upload frame to Minio in a separate thread."""
        if not self.minio_client:
            return

        try:
            # Encode frame to jpg
            _, buffer = cv2.imencode(".jpg", frame)
            data = BytesIO(buffer)
            
            # Generate object name with timestamp
            object_name = f"frame_{int(time.time() * 1000)}.jpg"
            
            self.minio_client.put_object(
                MINIO_BUCKET_NAME,
                object_name,
                data,
                len(buffer),
                content_type="image/jpeg",
            )
            logger.debug(f"Uploaded frame to Minio: {object_name}") # Commented out to reduce noise
        except Exception as e:
            logger.error(f"Failed to upload frame to Minio: {e}")

    def camera_thread(self):
        """
        Continuously capture frames from the RTSP stream and push
        them into frame_queue.  Drops oldest frame when queue is full.
        """
        logger.info(f"Camera thread started – connecting to {CAMERA_STREAM_URL}")
        cap = cv2.VideoCapture(CAMERA_STREAM_URL)

        if not cap.isOpened():
            logger.error("Failed to open camera stream, retrying in 3 s...")
            time.sleep(3)
            cap = cv2.VideoCapture(CAMERA_STREAM_URL)
            if not cap.isOpened():
                logger.error("Camera stream unavailable – camera thread exiting.")
                return

        logger.info("Camera stream opened successfully.")
        prev_time = time.monotonic()
        last_upload_time = 0
        frame_count = 0
        consecutive_errors = 0

        while not self._shutdown.is_set():
            ret, frame = cap.read()
            if not ret:
                consecutive_errors += 1
                logger.warning(
                    f"Camera read failed (attempt {consecutive_errors})"
                )
                if consecutive_errors >= 10:
                    logger.warning("Too many camera errors, attempting reconnect...")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(CAMERA_STREAM_URL)
                    consecutive_errors = 0
                time.sleep(0.1)
                continue

            consecutive_errors = 0
            frame_count += 1

            # Minio Upload (every MINIO_UPLOAD_INTERVAL seconds)
            current_time = time.time()
            if current_time - last_upload_time >= MINIO_UPLOAD_INTERVAL:
                threading.Thread(
                    target=self._upload_frame_to_minio,
                    args=(frame.copy(),), # Pass a copy to avoid race conditions if frame is modified
                    daemon=True
                ).start()
                last_upload_time = current_time

            # FPS calculation (every 1 second)
            now = time.monotonic()
            elapsed = now - prev_time
            if elapsed >= 1.0:
                self._fps = frame_count / elapsed
                frame_count = 0
                prev_time = now

            # Push frame, drop oldest if full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
            try:
                self.frame_queue.put_nowait(frame)
            except Full:
                pass

        cap.release()
        logger.info("Camera thread stopped.")

    # ── Thread 2: Vision processing ───────────────────────────────────────

    def vision_thread(self):
        """
        Read frames from frame_queue, run object detection, and enqueue
        robot commands when a detection is stable and the robot is IDLE.
        """
        logger.info("Vision thread started.")

        while not self._shutdown.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except Empty:
                continue

            annotated_frame, detections = self.object_detector.detect(frame)
            if annotated_frame is None:
                annotated_frame = frame

            # ── Detection stability logic ──
            if detections:
                detections.sort(key=lambda x: x.confidence, reverse=True)
                target_bbox = detections[0]

                self.detection_counter = self.detection_counter + 1
                logger.debug(
                    f"Detection: {target_bbox.class_name} "
                    f"(conf={target_bbox.confidence:.2f}) – "
                    f"stability {self.detection_counter}/{DETECTION_STABILITY_THRESHOLD}"
                )

                # Check if we should issue a command
                if (
                    self.detection_counter >= DETECTION_STABILITY_THRESHOLD
                    and self.robot_state == "IDLE"
                    and not self.target_locked
                ):
                    object_class = target_bbox.class_name

                    # Build the pickup/disposal command sequence
                    task_commands = self._build_task_commands(
                        target_bbox, object_class
                    )

                    if task_commands:
                        logger.info(
                            f"Target locked on '{object_class}' – "
                            f"enqueueing {len(task_commands)} commands"
                        )
                        self.target_locked = True
                        self.robot_state = "MOVING"
                        self.detection_counter = 0

                        # Send Kafka event
                        self.producer.send(
                            "robot_events",
                            {
                                "robot_id": self.robot_id,
                                "event_type": "pickup_start",
                                "object_class": object_class,
                                "details": "Starting pickup sequence",
                            },
                        )

                        # Enqueue the whole task as a list of commands
                        try:
                            self.cmd_queue.put_nowait(
                                {
                                    "commands": task_commands,
                                    "object_class": object_class,
                                }
                            )
                        except Full:
                            logger.warning(
                                "cmd_queue full – dropping task, "
                                "robot will return to IDLE"
                            )
                            self.robot_state = "IDLE"
                            self.target_locked = False
            else:
                # No detection – reset stability counter
                if self.detection_counter > 0:
                    logger.debug("Detection lost – resetting stability counter")
                self.detection_counter = 0

            # ── Overlay HUD on the frame ──
            state = self.robot_state
            cv2.putText(
                annotated_frame,
                f"State: {state}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if state == "IDLE" else (0, 0, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"FPS: {self._fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            if self.target_locked:
                cv2.putText(
                    annotated_frame,
                    "TARGET LOCKED",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # Store latest frame for display by the main thread
            with self._frame_lock:
                self._display_frame = annotated_frame

        logger.info("Vision thread stopped.")

    def _build_task_commands(self, target_bbox, object_class):
        """
        Build a list of serial command strings that constitute a full
        pickup → disposal → return sequence.

        Returns a list of command strings or None if not possible.
        """
        if not self.servo_control:
            logger.warning("Servo control not initialized – cannot build commands")
            return None

        commands = []

        # 1. Center the object (compute base servo target)
        center_x, _ = target_bbox.center
        frame_center_x = self.frame_width / 2
        distance_from_center = center_x - frame_center_x

        if abs(distance_from_center) > self.center_threshold:
            # Estimate how many degrees to adjust (rough mapping)
            if distance_from_center > 0:
                new_base = max(0, self.servo_control.basePosition - 1)
            else:
                new_base = min(180, self.servo_control.basePosition + 1)
            commands.append(f"S1A{new_base}")
            self.servo_control.basePosition = new_base

        # 2. Get distance command
        commands.append("DIST")

        # 3. Open gripper before reaching
        commands.append("S5A100")

        # 4. Move to pickup position (use a default middle distance)
        #    The robot thread will handle DIST response and adjust if needed
        #    For now we use "20" as a reasonable default pickup distance
        if "20" in self.servo_control.positionData:
            commands.append(self.servo_control.positionData["20"])
        elif "21" in self.servo_control.positionData:
            commands.append(self.servo_control.positionData["21"])

        # 5. Close gripper
        commands.append("S5A40")

        # 6. Move to disposal position
        disposal_map = {
            "paper": "paperDisposal",
            "metal": "metalDisposal",
            "plastic": "plasticDisposal",
        }
        disposal_key = disposal_map.get(object_class, "paperDisposal")
        if disposal_key in self.servo_control.positionData:
            commands.append(self.servo_control.positionData[disposal_key])

        # 7. Open gripper to release
        commands.append("S5A100")

        # 8. Return to rest / neutral
        if "rest" in self.servo_control.positionData:
            commands.append(self.servo_control.positionData["rest"])
        elif "neutral" in self.servo_control.positionData:
            commands.append(self.servo_control.positionData["neutral"])

        return commands

    # ── Thread 3: Robot serial control ────────────────────────────────────

    def robot_thread(self):
        """
        Pop tasks from cmd_queue, send each command over serial,
        wait for OK, then advance to the next command.
        Only thread allowed to touch the serial port.
        """
        logger.info("Robot thread started.")

        while not self._shutdown.is_set():
            try:
                task = self.cmd_queue.get(timeout=1.0)
            except Empty:
                continue

            commands = task["commands"]
            object_class = task["object_class"]
            logger.info(
                f"Robot executing task: {len(commands)} commands "
                f"for '{object_class}'"
            )

            success = True
            for i, cmd in enumerate(commands):
                if self._shutdown.is_set():
                    break

                # Special handling for DIST command
                if cmd == "DIST":
                    logger.debug("Requesting distance measurement...")
                    if self.servo_control:
                        self.servo_control.ser.write("DIST\n".encode("utf-8"))
                        # Wait for DISTC response
                        dist_start = time.monotonic()
                        distance = None
                        while (time.monotonic() - dist_start) < 5.0:
                            if self.servo_control.ser.in_waiting > 0:
                                line = (
                                    self.servo_control.ser.readline()
                                    .decode("utf-8")
                                    .strip()
                                )
                                logger.debug(f"Serial RX: {line}")
                                if line.startswith("DISTC"):
                                    try:
                                        distance = int(line[5:])
                                        logger.info(
                                            f"Distance measured: {distance}cm"
                                        )
                                        self.producer.send(
                                            "robot_telemetry",
                                            {
                                                "robot_id": self.robot_id,
                                                "distance": distance,
                                            },
                                        )
                                    except ValueError:
                                        logger.warning(
                                            f"Bad distance response: {line}"
                                        )
                                    break
                            time.sleep(0.05)

                        # If we got a valid distance, swap the next pickup
                        # position command with the correct one
                        if distance is not None and self.servo_control:
                            dist_str = str(int(distance) + 1)
                            pos_data = self.servo_control.positionData
                            if dist_str in pos_data:
                                # Replace the next command (pickup position) if exists
                                if i + 2 < len(commands):
                                    # i+1 = open gripper, i+2 = pickup position
                                    commands[i + 2] = pos_data[dist_str]
                                    logger.debug(
                                        f"Updated pickup position to "
                                        f"distance {dist_str}cm"
                                    )
                            else:
                                # Find nearest
                                nearest = self._find_nearest_position(
                                    distance + 1
                                )
                                if (
                                    nearest
                                    and i + 2 < len(commands)
                                ):
                                    commands[i + 2] = pos_data[nearest]
                                    logger.debug(
                                        f"Updated pickup position to "
                                        f"nearest {nearest}cm"
                                    )
                    continue  # DIST is not a servo command

                # Normal servo command — send and wait for OK
                logger.debug(f"Command [{i+1}/{len(commands)}]: {cmd}")
                if self.servo_control:
                    ok = self.servo_control.send_command(cmd)
                    if not ok:
                        logger.warning(
                            f"Command failed (no OK): {cmd} – aborting task"
                        )
                        success = False
                        break
                else:
                    logger.warning(
                        "No servo control – simulating command execution"
                    )
                    time.sleep(0.5)

            # ── Task complete ──
            if success:
                logger.info(
                    f"Task completed successfully for '{object_class}'"
                )
                self.producer.send(
                    "robot_events",
                    {
                        "robot_id": self.robot_id,
                        "event_type": "pickup_success",
                        "object_class": object_class,
                        "details": "Pickup and disposal completed",
                    },
                )
            else:
                logger.warning(f"Task failed for '{object_class}'")
                self.producer.send(
                    "robot_events",
                    {
                        "robot_id": self.robot_id,
                        "event_type": "pickup_fail",
                        "object_class": object_class,
                        "details": "Pickup sequence failed",
                    },
                )

            # Release lock and go IDLE
            self.robot_state = "IDLE"
            self.target_locked = False
            logger.info("Robot returned to IDLE – ready for next task")

        logger.info("Robot thread stopped.")

    def _find_nearest_position(self, distance):
        """Find the nearest available position for a given distance."""
        try:
            distance = int(distance)
            positions_int = {}
            for pos in self.servo_control.positionData.keys():
                try:
                    pos_int = int(pos)
                    positions_int[pos] = pos_int
                except ValueError:
                    continue

            if not positions_int:
                return None

            closest_pos = min(
                positions_int.keys(),
                key=lambda x: abs(positions_int[x] - distance),
            )

            if abs(positions_int[closest_pos] - distance) <= 30:
                return closest_pos
            return None
        except Exception as e:
            logger.error(f"Error finding nearest position: {e}")
            return None

    # ── Main entry point ──────────────────────────────────────────────────

    def run(self):
        """
        Start all threads and run the OpenCV display loop on the main thread.
        Press 'q' to quit.
        """
        logger.info("Starting robotic arm controller (multi-threaded)...")

        # Start daemon threads
        threads = [
            threading.Thread(
                target=self.camera_thread, name="CameraThread", daemon=True
            ),
            threading.Thread(
                target=self.vision_thread, name="VisionThread", daemon=True
            ),
            threading.Thread(
                target=self.robot_thread, name="RobotThread", daemon=True
            ),
        ]

        for t in threads:
            logger.info(f"Starting {t.name}...")
            t.start()

        logger.info("All threads started. Press 'q' to quit.")

        try:
            while not self._shutdown.is_set():
                # Display the latest annotated frame from the vision thread
                with self._frame_lock:
                    frame = self._display_frame

                if frame is not None:
                    cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed – shutting down...")
                    break

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt – shutting down...")
        finally:
            self._shutdown.set()
            logger.info("Waiting for threads to finish...")

            # Give threads a moment to exit their loops
            for t in threads:
                t.join(timeout=3.0)

            cv2.destroyAllWindows()

            if self.servo_control:
                try:
                    self.servo_control.close()
                except Exception as e:
                    logger.error(f"Error closing servo control: {e}")

            logger.info("Robotic arm controller stopped.")
