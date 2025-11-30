"""
Module for object detection using YOLO models.
"""

from ultralytics import YOLO

from src.config.config import CONFIDENCE_THRESHOLD, MODEL_PATH, TRACKING_ENABLED
from src.detector.bbox_utils import BoundingBox


class ObjectDetector:
    """
    A class for detecting objects using YOLO models.
    """

    def __init__(self, model_path=None, conf=None, tracking=None):
        """
        Initialize the object detector with a YOLO model.

        Args:
            model_path (str, optional): Path to the YOLO model weights.
                                       Defaults to path in config.
            conf (float, optional): Confidence threshold for detections.
                                   Defaults to threshold in config.
            tracking (bool, optional): Whether to enable object tracking.
                                      Defaults to setting in config.
        """
        self.model_path = model_path if model_path is not None else MODEL_PATH
        self.conf = conf if conf is not None else CONFIDENCE_THRESHOLD
        self.tracking = tracking if tracking is not None else TRACKING_ENABLED

        self.model = self._load_model()

    def _load_model(self):
        """
        Load the YOLO model from the specified path.

        Returns:
            YOLO: The loaded YOLO model
        """
        try:
            model = YOLO(self.model_path)
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model from {self.model_path}: {str(e)}"
            )

    def detect(self, frame, verbose=False):
        """
        Detect objects in a frame.

        Args:
            frame (numpy.ndarray): The input frame
            verbose (bool, optional): Whether to print verbose output

        Returns:
            tuple: (annotated_frame, detections)
                - annotated_frame: Frame with detection annotations
                - detections: List of BoundingBox objects
        """
        if frame is None:
            return None, []

        if self.tracking:
            results = self.model.track(
                frame, persist=True, verbose=verbose, conf=self.conf
            )
        else:
            results = self.model.predict(frame, verbose=verbose, conf=self.conf)

        if not results or len(results) == 0:
            return frame, []

        detections = []
        result = results[0]

        annotated_frame = result.plot()

        if result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                box = result.boxes[i]

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                try:
                    class_id = int(box.cls[0].item())
                    class_name = result.names[class_id]
                    confidence = box.conf[0].item()
                except (IndexError, KeyError):
                    class_name = "unknown"
                    confidence = 0.0

                bbox = BoundingBox(x1, y1, x2, y2, class_name, confidence)
                detections.append(bbox)

        return annotated_frame, detections
