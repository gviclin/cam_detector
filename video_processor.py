"""
Video stream processing and preprocessing module
"""
import cv2
import numpy as np
from typing import Tuple
import config


class VideoProcessor:
    """Video stream processing manager"""
    
    @staticmethod
    def crop_sides(img: np.ndarray, target_ratio: float = 4/3) -> np.ndarray:
        """
        Crop image horizontally to achieve target aspect ratio
        
        Args:
            img: BGR or RGB image (H, W, C)
            target_ratio: Target width/height ratio
            
        Returns:
            Cropped image with target ratio
        """
        h, w = img.shape[:2]
        current_ratio = w / h

        if current_ratio <= target_ratio:
            return img
        
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        cropped = img[:, left:left+new_w]
        return cropped
    
    @staticmethod
    def preprocess_frame(frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for detection
        
        Args:
            frame: Raw camera frame
            
        Returns:
            Preprocessed and resized frame
        """

        # h, w = frame.shape[:2]

        # print(f"Original frame size {h}x{w}")
        # # Crop sidesx
        # frame = VideoProcessor.crop_sides(frame, target_ratio=config.TARGET_RATIO)
        
        # # Resize
        # frame = cv2.resize(frame, (config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
        
        return frame
    
    @staticmethod
    def draw_detection(img: np.ndarray, bbox: Tuple[int, int, int, int], 
                      label: str, color: Tuple[int, int, int] = (0, 255, 0)) -> None:
        """
        Draw detection box on image
        
        Args:
            img: Image to draw on
            bbox: Coordinates (x1, y1, x2, y2)
            label: Label to display
            color: BGR box color
        """
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2)


class RTSPCapture:
    """RTSP capture manager with automatic reconnection"""
    
    def __init__(self, url: str, camera_name: str):
        """
        Initialize RTSP capture
        
        Args:
            url: RTSP camera URL
            camera_name: Camera name (for logs)
        """
        self.url = url
        self.camera_name = camera_name
        self.cap = None
        self._connect()
    
    def _connect(self):
        """Establish RTSP stream connection"""
        print(f"[{self.camera_name}] Connecting to RTSP stream...")
        self.cap = cv2.VideoCapture(self.url)
        if self.cap.isOpened():
            print(f"[{self.camera_name}] ✅ Connection established")
            # Si un FPS est fourni, on le change
            # fps = 2
            if fps is not None:
                success = self.cap.set(cv2.CAP_PROP_FPS, fps)
                if success:
                    print(f"[{self.camera_name}] 🎯 FPS set to {fps}")
                else:
                    print(f"[{self.camera_name}] ⚠️ Failed to set FPS")

            # Changer la résolution si demandée
            # width = 640
            # height = 640
            if width is not None and height is not None:
                success_w = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                success_h = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                if success_w and success_h:
                    print(f"[{self.camera_name}] 🎯 Resolution set to {width}x{height}")
                else:
                    print(f"[{self.camera_name}] ⚠️ Failed to set resolution")


        else:
            print(f"[{self.camera_name}] ⚠️ Connection failed")
    
    def read(self) -> Tuple[bool, np.ndarray]:
        """
        Read frame from stream
        
        Returns:
            Tuple (success, frame)
        """
        if not self.cap or not self.cap.isOpened():
            self._reconnect()
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            self._reconnect()
            return False, None
        
        return True, frame
    
    def _reconnect(self):
        """Reconnect to stream on error"""
        print(f"[{self.camera_name}] Stream interrupted, reconnecting...")
        if self.cap:
            self.cap.release()
        
        import time
        time.sleep(2)
        self._connect()
    
    def release(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        print(f"[{self.camera_name}] Capture released")