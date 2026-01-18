"""
Video stream processing and preprocessing module
"""
import os
import cv2
import signal
import numpy as np
from typing import Tuple
import config
import subprocess
# import atexit

# import sys



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
        self.proc = None

        self._connect()

        # # cleanup automatique à la fermeture du script
        # atexit.register(self.release)


    
    def _connect(self):
        """Establish RTSP stream connection"""
        print(f"[{self.camera_name}] Connecting to RTSP stream...")

        self.width,  self.height = self.get_resolution(self.url)

        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", self.url,
            "-vf", f"fps={config.VIDEO_FPS}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**7,
            preexec_fn=os.setsid  # crée un nouveau groupe pour tuer ffmpeg et ses enfants
        )

        self.frame_size = self.width * self.height * 3
        print(f"[{self.camera_name}] ✅ RTSPCapture configured !")
    
    def read(self) -> Tuple[bool, np.ndarray]:
        if not self.proc or self.proc.poll() is not None:
            return False, None

        raw = self.proc.stdout.read(self.frame_size)
        if not raw or len(raw) < self.frame_size:
            return False, None

        frame = np.frombuffer(raw, np.uint8)\
                .reshape((self.height, self.width, 3))\
                .copy()

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
        if not self.proc:
            return

        if self.proc.poll() is None: # Process not ended yet
            print(f"[{self.camera_name}] Stopping ffmpeg ...")
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=5)
                
            except subprocess.TimeoutExpired:
                print(f"[{self.camera_name}] ffmpeg force kill")
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)

            self.proc = None


        print(f"[{self.camera_name}] Capture released")

    def get_resolution(self, rtsp_url):
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            rtsp_url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"get_resolution { result.stdout.strip()} - {rtsp_url}")
        width, height = map(int, result.stdout.strip().split(','))
        return width, height