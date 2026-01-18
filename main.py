"""
Main application for multi-camera surveillance with YOLO detection
"""
import sys, os
import traceback
import signal
import threading
from threading import Thread, Event
from typing import List
import config
from mqtt_manager import MQTTManager
from detector import YOLODetector
from camera_monitor import CameraMonitor


class MultiCameraSystem:
    """Multi-camera surveillance system"""

    def __init__(self):
        print("=" * 60)
        print("🎬 Multi-Camera Surveillance System")
        print("=" * 60)

        # MQTT
        self.mqtt = MQTTManager()
        self.mqtt.connect()

        # YOLO
        self.detector = YOLODetector(device="intel:cpu")

        # Stop event partagé
        self.stop_event = Event()

        # gestion CTRL+C / kill
        # signal.signal(signal.SIGINT, self._signal_handler)
        # signal.signal(signal.SIGTERM, self._signal_handler)  

        # Cameras
        self.monitors: List[CameraMonitor] = []
        for cam_config in config.CAMERAS:
            self.monitors.append(
                CameraMonitor(
                    cam_config,
                    self.detector,
                    self.mqtt,
                    self.stop_event
                )
            )

        self.threads: List[Thread] = []

        print(f"✅ {len(self.monitors)} camera(s) configured")
        print("=" * 60)
        

    def start(self):
        """Start surveillance for all cameras"""

        # print(f"[Main Thread ] {threading.current_thread().name}, ID : {threading.get_ident()}")

        for monitor in self.monitors:
            t = Thread(
                target=monitor.run,
                name=f"Monitor-{monitor.name}",
                daemon=True
            )
            t.start()
            self.threads.append(t)
            # print(f"▶️  Thread started for {monitor.name}")

        print("\n🟢 System operational - Press Ctrl+C to stop")

        try:
            # Boucle d’attente non bloquante
            while not self.stop_event.is_set():
                for t in self.threads:
                    t.join(timeout=0.5)

        except KeyboardInterrupt:
            print("\n\n🛑 Stop requested (Ctrl+C)")
            self.stop()
        finally:
            self.restore_terminal()
            print("✅ Restaure terminal")

    def stop(self):
        """Stop system cleanly"""

        if self.stop_event.is_set():
            return  # évite double stop

        self.stop_event.set()

        # Stop cameras
        for monitor in self.monitors:
            monitor.stop()

        # Wait threads
        print("⏳ Waiting for threads...")
        for t in self.threads:
            t.join(timeout=2)

        # Stop detector
        if self.detector:
            self.detector.stop()
            self.detector = None

        # MQTT
        print("🔄 Disconnecting MQTT...")
        self.mqtt.disconnect()

        print("=" * 60)
        print("🎬 Multi-Camera Surveillance System stopped cleanly")
        print("=" * 60)

    # def _signal_handler(self, signum, frame):
    #         print(f"\n\n🛑 [maint thread] Stop requested (Ctrl+C). Signal {signum} received")
    #         self.stop()
    #         sys.exit(0)

    def restore_terminal(this):
        """Restaure bash à coup sûr"""
        os.system("stty sane")

def main():
    try:
        system = MultiCameraSystem()
        system.start()
    except Exception as e:
        exc_type, exc_value, tb = sys.exc_info()
        filename = tb.tb_frame.f_code.co_filename
        lineno = tb.tb_lineno
        print(f"\n❌ {filename}:{lineno} Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
