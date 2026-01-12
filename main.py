"""
Main application for multi-camera surveillance with YOLO detection
"""
import sys
from threading import Thread
from typing import List
import config
from mqtt_manager import MQTTManager
from detector import YOLODetector
from camera_monitor import CameraMonitor
from threading import Event


class MultiCameraSystem:
    """Multi-camera surveillance system"""
    
    def __init__(self):
        """Initialize system"""
        print("="*60)
        print("🎬 Multi-Camera Surveillance System")
        print("="*60)
        
        # Initialize MQTT
        self.mqtt = MQTTManager()
        self.mqtt.connect()
        
        # Initialize YOLO detector
        self.detector = YOLODetector(device="intel:cpu")

        self.stop_event = Event()
        
        # Create camera monitors
        self.monitors: List[CameraMonitor] = []
        for cam_config in config.CAMERAS:
            monitor = CameraMonitor(
                cam_config,
                self.detector,
                self.mqtt,
                self.stop_event
            )
            self.monitors.append(monitor)        
        
        print(f"✅ {len(self.monitors)} camera(s) configured")
        print("="*60)
    
    def start(self):
        """Start surveillance for all cameras"""
        threads = []
        
        for monitor in self.monitors:
            thread = Thread(
                target=monitor.run,
                name=f"Monitor-{monitor.name}",
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
            print(f"▶️  Thread started for {monitor.name}")
        
        print("\n🟢 System operational - Press Ctrl+C to stop")
        

        try:
            # On attend que les threads tournent, mais avec timeout pour Ctrl+C
            while thread in threads:
                thread.join(timeout=1)
        except KeyboardInterrupt:
             # Ctrl+C détecté → déclenche l'arrêt propre
            print("\n\n🛑 Stop requested...")
            self.stop_event.set() 
            self.stop()
            for thread in threads:
                thread.join()
    
    def stop(self):
        """Stop system cleanly"""
        print("🔄 Stopping monitors...")
        for monitor in self.monitors:
            monitor.stop()
        
        print("🔄 Disconnecting MQTT...")
        self.mqtt.disconnect()
        
        print("✅ System stopped cleanly")


def main():
    """Application entry point"""
    try:
        system = MultiCameraSystem()
        system.start()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()