import sys
import os
import cv2
import time
import json

# Add src to sys.path to allow imports from the src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "src"))

try:
    from config.config import CAMERA_STREAM_URL, SERVO_PORT, LOG_LEVEL
    from controller.servo_controller import ServoControl
    from detector.object_detector import ObjectDetector
    from utils.logger import get_logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

logger = get_logger(__name__)

# Global state for mouse callback (using list for mutability)
_clicked_x = [None]
_prev_clicked_x = [None]


def mouse_callback(event, x, y, flags, param):
    global _clicked_x, _prev_clicked_x
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicked_x[0] = x
        _prev_clicked_x[0] = x


def main():
    # Initialize servo controller
    servo = None
    try:
        logger.info(f"Connecting to servo controller on {SERVO_PORT}...")
        servo = ServoControl(COM=SERVO_PORT)
        logger.info("Servo controller initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to servo controller: {e}")
        print("Warning: Running without servo control (hardware may not be connected).")

    # Initialize object detector for frame processing
    try:
        detector = ObjectDetector()
        logger.info("Object detector initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize object detector: {e}")
        detector = None

    # Open camera stream
    logger.info(f"Opening camera stream: {CAMERA_STREAM_URL}")
    cap = cv2.VideoCapture(CAMERA_STREAM_URL)

    # Check if stream is opened, otherwise try default camera
    if not cap.isOpened():
        logger.warning(
            f"Failed to open camera stream: {CAMERA_STREAM_URL}. Trying default camera (0)..."
        )
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open any camera source.")
            return

    initial_position = None
    saved_position = None
    clicked_x = None
    prev_clicked_x = None

    # We'll use the servo's internal basePosition tracker if available
    # Otherwise we'll maintain it locally
    current_s1_angle = 90
    if servo:
        current_s1_angle = servo.basePosition

    print("\n" + "=" * 40)
    print("      ROBOTIC ARM CAMERA CONTROL")
    print("=" * 40)
    print("Controls:")
    print("  LEFT ARROW  : Move S1 Left")
    print("  RIGHT ARROW : Move S1 Right")
    print("  CLICK       : Mark position & draw line")
    print("  'w'         : Mark Initial Position")
    print("  's'         : Save Current Position (with pixel values)")
    print("  'q'         : Quit")
    print("=" * 40 + "\n")

    cv2.namedWindow("Robotic Arm Control Feed")
    cv2.setMouseCallback("Robotic Arm Control Feed", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to receive frame from camera")
            break

        # Check for mouse click and update position
        if _clicked_x[0] is not None and _clicked_x[0] != prev_clicked_x:
            clicked_x = _clicked_x[0]
            initial_position = current_s1_angle
            prev_clicked_x = clicked_x
            print(
                f"[*] Clicked at x={clicked_x}, marked initial position: {initial_position}"
            )
            logger.info(f"User clicked at x={clicked_x}, S1 angle: {initial_position}")

        # Reuse existing ObjectDetector to draw boxes and labels
        if detector:
            annotated_frame, detections = detector.detect(frame)
        else:
            annotated_frame = frame

        # Overlay status information
        overlay_text = [
            f"S1 Current Angle: {current_s1_angle}",
            f"Initial Pos: {initial_position if initial_position is not None else 'Not set'}",
            f"Saved Pos: {saved_position if saved_position is not None else 'Not set'}",
        ]

        for i, text in enumerate(overlay_text):
            cv2.putText(
                annotated_frame,
                text,
                (10, 30 + (i * 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Get frame dimensions for drawing lines
        frame_height, frame_width = annotated_frame.shape[:2]
        center_x = frame_width // 2

        # Draw center line (green) - always visible
        cv2.line(
            annotated_frame, (center_x, 0), (center_x, frame_height), (0, 255, 0), 2
        )

        # Draw user clicked line (red) - only when clicked
        if clicked_x is not None:
            cv2.line(
                annotated_frame,
                (clicked_x, 0),
                (clicked_x, frame_height),
                (0, 0, 255),
                2,
            )

        cv2.imshow("Robotic Arm Control Feed", annotated_frame)

        # waitKeyEx(1) to capture special keys like arrow keys
        # Standard OpenCV key codes for Windows:
        # Left: 2424832, Right: 2555904
        # Linux/Mac: Left=65361, Right=65363
        key = cv2.waitKeyEx(1)

        if key == ord("q"):
            break
        elif key == ord("w"):
            initial_position = current_s1_angle
            print(f"[*] Marked initial position: {initial_position}")
        elif key == ord("s"):
            saved_position = current_s1_angle
            print(f"[*] Saved new position: {saved_position}")

            # Clear the manual line after saving
            clicked_x_to_save = clicked_x
            clicked_x = None
            _clicked_x[0] = None

            # Save to JSON file for persistence
            try:
                save_data = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "center_x": center_x,
                    "clicked_x": clicked_x_to_save,
                    "initial_s1_angle": initial_position,
                    "saved_s1_angle": saved_position,
                }
                json_file = os.path.join(current_dir, "saved_positions.json")
                # Read existing data or create empty list
                existing_data = []
                if os.path.exists(json_file):
                    with open(json_file, "r") as f:
                        try:
                            existing_data = json.load(f)
                        except json.JSONDecodeError:
                            existing_data = []
                # Append new data
                existing_data.append(save_data)
                # Write back
                with open(json_file, "w") as f:
                    json.dump(existing_data, f, indent=2)
                print(
                    f"[*] Saved to JSON: center_x={center_x}, clicked_x={clicked_x_to_save}, initial_s1={initial_position}, saved_s1={saved_position}"
                )
            except Exception as e:
                logger.error(f"Failed to save position to JSON file: {e}")

        # Handle Arrow Keys
        # Left Arrow: Windows=2424832, Linux/Mac=65361
        # Right Arrow: Windows=2555904, Linux/Mac=65363
        elif key in (2424832, 65361):
            if servo:
                servo.baseServoLeft()
                current_s1_angle = servo.basePosition
            else:
                current_s1_angle = min(180, current_s1_angle + 1)
            print(f"Moving S1 Left -> {current_s1_angle}")

        # Right Arrow
        elif key in (2555904, 65363):
            if servo:
                servo.baseServoRight()
                current_s1_angle = servo.basePosition
            else:
                current_s1_angle = max(0, current_s1_angle - 1)
            print(f"Moving S1 Right -> {current_s1_angle}")

    # Cleanup
    logger.info("Closing camera control...")
    cap.release()
    cv2.destroyAllWindows()
    if servo:
        servo.close()


if __name__ == "__main__":
    main()
