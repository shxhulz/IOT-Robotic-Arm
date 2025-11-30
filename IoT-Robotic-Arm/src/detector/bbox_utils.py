"""
Utilities for working with bounding boxes and geometric calculations.
"""

import math


class BoundingBox:
    """
    A class representing a bounding box with top-left and bottom-right coordinates.
    """

    def __init__(self, x1, y1, x2, y2, class_name=None, confidence=None):
        """
        Initialize a bounding box with its coordinates and optional metadata.

        Args:
            x1 (float): X coordinate of top-left corner
            y1 (float): Y coordinate of top-left corner
            x2 (float): X coordinate of bottom-right corner
            y2 (float): Y coordinate of bottom-right corner
            class_name (str, optional): Class name of detected object
            confidence (float, optional): Confidence score of detection
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.class_name = class_name
        self.confidence = confidence

    @property
    def center(self):
        """
        Returns the center coordinates of the bounding box.

        Returns:
            tuple: (center_x, center_y)
        """
        center_x = (self.x1 + self.x2) / 2
        center_y = (self.y1 + self.y2) / 2
        return (center_x, center_y)

    @property
    def width(self):
        """
        Returns the width of the bounding box.

        Returns:
            float: Width
        """
        return self.x2 - self.x1

    @property
    def height(self):
        """
        Returns the height of the bounding box.

        Returns:
            float: Height
        """
        return self.y2 - self.y1

    def distance_to_point(self, x, y):
        """
        Calculate the distance from a point to this bounding box.

        Args:
            x (float): X coordinate of the point
            y (float): Y coordinate of the point

        Returns:
            float: The shortest distance from the point to the box
        """
        closest_x = max(self.x1, min(x, self.x2))
        closest_y = max(self.y1, min(y, self.y2))

        dx = x - closest_x
        dy = y - closest_y

        return math.sqrt(dx * dx + dy * dy)

    def __str__(self):
        """String representation of the bounding box."""
        return f"BBox({self.x1:.1f}, {self.y1:.1f}, {self.x2:.1f}, {self.y2:.1f}), class: {self.class_name}, conf: {self.confidence:.2f if self.confidence else None}"

    @classmethod
    def from_yolo_result(cls, detection):
        """
        Create a BoundingBox from a YOLO detection result.

        Args:
            detection: A detection result from YOLO model

        Returns:
            BoundingBox: A new BoundingBox instance
        """
        box = detection.boxes.xyxy[0].tolist()
        try:
            class_id = int(detection.boxes.cls[0].item())
            class_name = detection.names[class_id]
            confidence = detection.boxes.conf[0].item()
        except (IndexError, KeyError):
            class_name = None
            confidence = None

        return cls(box[0], box[1], box[2], box[3], class_name, confidence)
