"""
Object detection module using YOLO
"""
# import time
# from datetime import datetime
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple
import config
from dataclasses import dataclass


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
    """YOLO detection manager with OpenVINO"""
    
    def __init__(self, model_path: str = None, device: str = "intel:cpu"):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to model (default from config)
            device: Compute device ("intel:cpu", "intel:gpu", "intel:npu")
        """
        self.model_path = model_path or config.MODEL_PATH
        self.device = device
        self.model = YOLO(self.model_path)
        print(f"✅ YOLO model loaded: {self.model_path} (device: {device})")
    
    def detect(self, image: np.ndarray, conf_threshold: float = None,
               target_classes: List[str] = None) -> List[Detection]:
        """
        Perform detection on image
        
        Args:
            image: Image to analyze (BGR)
            conf_threshold: Confidence threshold (default from config)
            target_classes: List of classes to detect (e.g. ["person", "car"])
            
        Returns:
            List of detections
        """
        conf_threshold = conf_threshold or config.CONF_THRESHOLD
        
        # Inference
        results = self.model(image, device=self.device, verbose=False)        
        
        detections = []
        
        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                class_id = int(cls)
                class_name = config.COCO_CLASSES[class_id]
                confidence = float(conf)
                
                # Filter by target class
                if target_classes and class_name not in target_classes:
                    continue
                
                # Filter by confidence
                if confidence < conf_threshold:
                    continue
                
                # Extract bbox
                x1, y1, x2, y2 = map(int, box)
                
                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2)
                )
                
                detections.append(detection)
        
        return detections
    
    def detect_persons(self, image: np.ndarray, conf_threshold: float = None) -> List[Detection]:
        """
        Detect only persons
        
        Args:
            name: name of the camera
            image: Image to analyze
            conf_threshold: Confidence threshold
            
        Returns:
            List of person detections
        """
        return self.detect(image, conf_threshold, target_classes=["person"])