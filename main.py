"""
Main application for multi-camera surveillance with YOLO detection
"""
import sys, os
import traceback
import signal
import time
import queue  # pour l'exception Empty
from multiprocessing import Queue, Process, Event
from typing import List
import config
from mqtt_manager import MQTTManager
from detector import YOLODetector
from camera_monitor import CameraMonitor

def monitor_cpu(stop_event):
    import psutil
    p = psutil.Process()
    while not stop_event.is_set():
        cpu_total = psutil.cpu_percent(interval=0.5)
        cpu_proc = p.cpu_percent(interval=None)
        print(f"[CPU] total={cpu_total:.1f}% | process={cpu_proc:.1f}%")
        time.sleep(0.5)  # ajustable


class MultiCameraSystem:
    """Multi-camera surveillance system"""

    def __init__(self):
        print("=" * 60)
        print("🎬 Multi-Camera Surveillance System")
        print("=" * 60)

        # MQTT
        self.mqtt = MQTTManager(on_enable_change=self.handle_detection_change)
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

        # self.threads: List[Thread] = []
        self.processes: List[Process] = []

        # Crée une queue partagée pour MQTT
        self.mqtt_queue = Queue()

        print(f"✅ {len(self.monitors)} camera(s) configured")
        print("=" * 60)
        

    def start(self):
        """Start surveillance for all cameras"""

        # print(f"[Main Thread ] {threading.current_thread().name}, ID : {threading.get_ident()}")

        # t = Process(
        #     target=monitor_cpu,
        #     args=(self.stop_event,),
        #     daemon=True)
        # t.start()

        # self.processes.append(t)

        self.stop_event.clear()

        for monitor in self.monitors:
            p = Process(
                    target=monitor.run,
                    args=(self.mqtt_queue,))
            p.start()
            self.processes.append(p)

        print("\n🟢 System operational - Press Ctrl+C to stop")

        while True:
            try:
                    camera_name, state = self.mqtt_queue.get(timeout=0.1)
                    self.mqtt.publish_state(camera_name, state)
            except queue.Empty:
                pass
            except KeyboardInterrupt:
                print("\n\n🛑 Stop requested (Ctrl+C)")
                self.stop()

        self.restore_terminal()
        print("✅ Restaure terminal")


    def stop(self):
        """Stop system cleanly"""

        if self.stop_event.is_set():
            return  # évite double stop

        self.stop_event.set()

        # Wait threads
        print("⏳ Waiting for threads...")
        for p in self.processes:
            p.join()        # attend qu'ils finissent proprement

        # Stop cameras
        for monitor in self.monitors:
            monitor.stop()

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



    def handle_detection_change(self, camera_name: str, enabled: bool):
        state = "ENABLED" if enabled else "DISABLED"
        # print(f"[APP] Detection {state} for camera: {camera_name}")

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
