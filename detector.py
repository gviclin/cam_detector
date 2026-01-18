"""
Object detection module using YOLO
"""
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple
import config
from dataclasses import dataclass
import threading
import gc


@dataclass
class Detection:
    """Represents an object detection"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2

    def get_label(self) -> str:
        """Return formatted label for display"""
        return f"{self.class_name} {self.confidence*100:.0f}%"


class YOLODetector:
    """YOLO detection manager"""

    def __init__(self, model_path: str = None, device: str = "intel:cpu"):
        """
        Initialize YOLO detector
        """
        self.model_path = model_path or config.MODEL_PATH
        self.device = device

        self._lock = threading.Lock()
        self._running = True

        print(f"🔄 Load YOLO model")
        self.model = YOLO(self.model_path)
        print(f"✅ YOLO model loaded: {self.model_path} (device: {device})")

    def stop(self):
        """
        Stop detector and release resources cleanly
        """
        print("🔄 Stopping YOLODetector...")
        with self._lock:
            self._running = False

            # Libération explicite (important pour OpenVINO / threads internes)
            if self.model is not None:
                del self.model
                self.model = None

        gc.collect()
        print("✅ YOLODetector stopped")

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = None,
        target_classes: List[str] = None
    ) -> List[Detection]:
        """
        Perform detection on image
        """
        with self._lock:
            if not self._running or self.model is None:
                return []

            conf_threshold = conf_threshold or config.CONF_THRESHOLD

            # Inference
            results = self.model(image, device=self.device, verbose=False)

        detections = []

        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                class_id = int(cls)
                class_name = config.COCO_CLASSES[class_id]
                confidence = float(conf)

                if target_classes and class_name not in target_classes:
                    continue
                if confidence < conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box)

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2)
                    )
                )

        return detections

    def detect_persons(
        self,
        image: np.ndarray,
        conf_threshold: float = None
    ) -> List[Detection]:
        """
        Detect only persons
        """
        return self.detect(
            image,
            conf_threshold,
            target_classes=["person"]
        )
