"""
Configuration file for the object detection and tracking system.
Contains paths, model parameters, and other settings.
"""
import os

LOG_LEVEL = "INFO"  # Set to "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL"

# Camera settings
CAMERA_STREAM_URL = "http://192.168.4.1/stream"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# YOLO model settings
MODEL_PATH = r"D:\RoboticArm\IOT-Robotic-Arm\IoT-Robotic-Arm\scripts\mlruns\1\11bd7dfa85604418addc2e9e9b7c4a73\artifacts\weights\best.pt"
CONFIDENCE_THRESHOLD = 0.4
TRACKING_ENABLED = True

# GUI settings
WINDOW_NAME = "Object Detection"
EXIT_KEY = "q"

# Servo control settings
SERVO_PORT = "COM4"
DETECTION_CONFIDENCE_THRESHOLD = 0.6
SERVO_MOVEMENT_COOLDOWN = 1  # seconds between movements
GRIPPER_SERVO = 5
GRIPPER_OPEN_ANGLE = 105
GRIPPER_CLOSE_ANGLE = 1
CENTER_THRESHOLD = 10  # Pixels from center to consider object centered
CENTERING_OFFSET_X = 10  # Pixels to shift desired centering target from frame center
CENTERING_STEP_DEGREES = 1  # Base-servo degrees per centering correction step
CENTERING_SETTLE_SECONDS = 1.0  # Wait time before post-centering verification
CENTERING_MAX_ADJUSTMENTS = 8  # Maximum centering correction attempts per task
CENTERING_VISION_TIMEOUT = 1.5  # Max seconds to wait for a fresh vision update

# Threading & Queue settings
FRAME_QUEUE_SIZE = 5  # Max frames buffered between camera and vision threads
CMD_QUEUE_SIZE = 5  # Max commands buffered between vision and robot threads
DETECTION_STABILITY_THRESHOLD = 5  # Consecutive frames with detection required before acting

# Serial communication
SERIAL_OK_TIMEOUT = 30  # Seconds to wait for OK response from Arduino

# Minio settings
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET_NAME = "camera-frames"
MINIO_UPLOAD_INTERVAL = 2  # Seconds between frame uploads
MINIO_SECURE = False  # Set to True if using HTTPS
