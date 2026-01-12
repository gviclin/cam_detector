
"""
Camera monitoring module with detection and alerts
"""
import os
import cv2
import time
import shutil
from datetime import datetime
from typing import Dict
import config
from video_processor import RTSPCapture, VideoProcessor
from detector import YOLODetector
from mqtt_manager import MQTTManager
from threading import Event


class CameraMonitor:
    """Surveillance monitor for a camera"""
    
    def __init__(self, camera_config: Dict, detector: YOLODetector, 
                 mqtt_manager: MQTTManager, stop_event: Event):
        """
        Initialize camera monitor
        
        Args:
            camera_config: Camera configuration (name, url)
            detector: YOLO detector instance
            mqtt_manager: MQTT manager instance
        """
        self.camera = camera_config
        self.name = camera_config["name"]
        self.detector = detector
        self.mqtt = mqtt_manager
        self.stop_event = stop_event
        
        # Detection state
        self.state = "OFF"
        self.last_detection_time = 0
        self.last_processed_time = 0
        self.frame_count = 0
        
        # Save directory
        self.save_dir = os.path.join(config.SAVE_DIR, self.name)
        self._prepare_save_directory()
        
        # Video capture
        self.capture = RTSPCapture(camera_config["url"], self.name)
    
    def _prepare_save_directory(self):
        """Prepare save directory (empty if exists)"""
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"[{self.name}] 📁 Directory ready: {self.save_dir}")
    
    def _should_process_frame(self) -> bool:
        """Determine if frame should be processed (throttling)"""
        now = time.time()
        if now - self.last_processed_time >= config.PROCESSING_INTERVAL:
            self.last_processed_time = now
            return True
        return False
    
    def _save_frame(self, frame, timestamp: str):
        """Save frame with detection"""
        # # Don't save for kitchen
        # if "kitchen" in self.name:
        #     return
        
        filename = f"{timestamp}-{self.frame_count:04d}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, frame)
    
    def _update_state(self, detected: bool):
        """Update detection state and publish on MQTT"""
        now = time.time()
        
        if detected:
            self.last_detection_time = now
            if self.state == "OFF":
                self.state = "ON"
                self.mqtt.publish_state(self.name, self.state)
        else:
            # Delay before switching to OFF
            if self.state == "ON" and (now - self.last_detection_time) > config.OFF_DELAY:
                self.state = "OFF"
                self.mqtt.publish_state(self.name, self.state)
    
    def process_frame(self, frame):
        """
        Process frame: detection, display and save
        
        Args:
            frame: Raw camera frame
        """
        self.frame_count += 1
        
        # Check if processing is enabled
        if not self.mqtt.is_detection_enabled():
            if self.state != "OFF":
                self.state = "OFF"
                self.mqtt.publish_state(self.name, self.state)
            return
        
        # Processing throttling
        if not self._should_process_frame():
            return
        
        # Preprocessing
        processed_frame = VideoProcessor.preprocess_frame(frame)
        
        # Detection
        start = time.time()
        detections = self.detector.detect_persons(processed_frame)
        # end = time.time()
        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        # print(f"{timestamp} - [{self.name}] Inference Duration : {(end - start) * 1000:.2f} ms")
        
        
        # Draw dectection bounding box        
        for det in detections:
            VideoProcessor.draw_detection(
                processed_frame,
                det.bbox,
                det.get_label()
            )
        
        # Update state and save
        detected = len(detections) > 0
        self._update_state(detected)
        
        if detected:
            self._save_frame(processed_frame, timestamp)
    
    def run(self):
        """Main monitoring loop"""
        print(f"[{self.name}] 🎥 Starting surveillance")
        
        # Publish initial state
        self.mqtt.publish_state(self.name, self.state)
        
        while not self.stop_event.is_set():
            ret, frame = self.capture.read()

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"{timestamp} - [{self.name}] new frame")
            
            if not ret:
                print(f"❌  {timestamp} - [{self.name}] Frame error")
                continue
            
            try:
                # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                # print(f"{timestamp} - process_frame")
                self.process_frame(frame)
            except Exception as e:
                print(f"[{self.name}] ❌ Processing error: {e}")
    
    def stop(self):
        """Stop surveillance"""
        self.capture.release()
        print(f"[{self.name}] 🛑 Surveillance stopped")