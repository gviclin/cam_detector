"""
RTSP streaming module via FFmpeg
"""
import subprocess
import numpy as np
from typing import Optional
import config


class RTSPStreamer:
    """RTSP streaming manager with FFmpeg"""
    
    def __init__(self, output_url: str = None, width: int = None, 
                 height: int = None, fps: int = None):
        """
        Initialize RTSP streamer
        
        Args:
            output_url: Output RTSP URL
            width: Frame width
            height: Frame height
            fps: Stream FPS
        """
        self.output_url = output_url or config.RTSP_OUTPUT_URL
        self.width = width or config.VIDEO_WIDTH
        self.height = height or config.VIDEO_HEIGHT
        self.fps = fps or config.VIDEO_FPS
        self.process: Optional[subprocess.Popen] = None
        
    def start(self):
        """Start FFmpeg process"""
        ffmpeg_cmd = [
            "ffmpeg",
            "-loglevel", "warning",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", config.FFMPEG_PRESET,
            "-tune", config.FFMPEG_TUNE,
            "-g", str(self.fps),
            "-keyint_min", str(self.fps),
            "-f", "rtsp",
            self.output_url
        ]
        
        try:
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"✅ RTSP streaming started: {self.output_url}")
        except Exception as e:
            print(f"❌ FFmpeg startup error: {e}")
            self.process = None
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Send frame to RTSP stream
        
        Args:
            frame: Frame to send (BGR)
            
        Returns:
            True if success, False otherwise
        """
        if not self.process or self.process.poll() is not None:
            print("⚠️ FFmpeg process not active")
            return False
        
        try:
            self.process.stdin.write(frame.tobytes())
            return True
        except Exception as e:
            print(f"❌ Frame write error: {e}")
            return False
    
    def stop(self):
        """Stop streaming"""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
                print("✅ RTSP streaming stopped")
            except Exception as e:
                print(f"⚠️ FFmpeg stop error: {e}")
                self.process.kill()
            finally:
                self.process = None
    
    def __enter__(self):
        """Context manager support"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.stop()