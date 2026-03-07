"""
Configuration file for the object detection and tracking system.
Contains paths, model parameters, and other settings.
"""

LOG_LEVEL = "DEBUG"  # Set to "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL"

# Camera settings
CAMERA_STREAM_URL = "http://192.168.4.1/stream"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# YOLO model settings
MODEL_PATH = "src/yolo/weights/best.pt"
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

# Threading & Queue settings
FRAME_QUEUE_SIZE = 5  # Max frames buffered between camera and vision threads
CMD_QUEUE_SIZE = 5  # Max commands buffered between vision and robot threads
DETECTION_STABILITY_THRESHOLD = 5  # Consecutive frames with detection required before acting

# Serial communication
SERIAL_OK_TIMEOUT = 30  # Seconds to wait for OK response from Arduino

# Minio settings
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET_NAME = "camera-frames"
MINIO_UPLOAD_INTERVAL = 2  # Seconds between frame uploads
MINIO_SECURE = False  # Set to True if using HTTPS
