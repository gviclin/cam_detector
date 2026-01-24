"""
MQTT manager for home automation system communication
"""
import paho.mqtt.client as mqtt
from datetime import datetime
from typing import Callable, Optional
import config
from multiprocessing import Manager


class MQTTManager:
    """MQTT communication manager"""
    
    def __init__(self, on_enable_change: Optional[Callable[[str, bool], None]] = None):
        """
        Initialize MQTT manager
        
        Args:
            on_enable_change: Callback called when enable state changes
        """
        self.mqtt_client = mqtt.Client(client_id="cam_detector")
        self.mqtt_client.username_pw_set(config.MQTT_USERNAME, config.MQTT_PASSWORD)

        # Link MQTT callbacks
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self._on_message

        self.on_enable_change = on_enable_change

        self.manager = Manager()

        self.detection_enabled = self.manager.dict({
            cam["name"]: False for cam in config.CAMERAS
        })

        # self.detection_enabled = {
        #     cam["name"]: False for cam in config.CAMERAS
        # }

        self.topic_to_camera = {
            cam["enable_topic"]: cam["name"] for cam in config.CAMERAS
        }


    def on_connect(self, client, userdata, flags, rc):
        if rc != 0:
            print(f"❌ MQTT connection failed (rc={rc})")
            return

        print(f"✅ Connected to MQTT broker {config.MQTT_BROKER}:{config.MQTT_PORT}")

        # Subscribe aux topics
        for cam in config.CAMERAS:
            client.subscribe(cam["enable_topic"])

        # Reset état côté HA
        for cam in config.CAMERAS:
                topic = cam["enable_topic"]
                self.detection_enabled[cam["name"]] = False
                self.mqtt_client.publish(topic, "OFF", retain=True)
        
    def _on_message(self, client, userdata, msg):
        """Callback for received MQTT messages"""
        payload = msg.payload.decode()
        topic = msg.topic

        if topic not in self.topic_to_camera:
            return

        camera_name = self.topic_to_camera[topic]

        if payload == "ON":
            self.detection_enabled[camera_name] = True
            print(f"🟢 Detection ENABLED for {camera_name}")
        elif payload == "OFF":
            self.detection_enabled[camera_name] = False
            print(f"🔴 Detection DISABLED for {camera_name}")
        
        if self.on_enable_change:
            self.on_enable_change(camera_name, self.detection_enabled[camera_name])
    
    def connect(self):
        """Connect to MQTT broker"""
        self.mqtt_client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)

        for cam in config.CAMERAS:
            self.mqtt_client.subscribe(cam["enable_topic"])

        self.mqtt_client.loop_start()

    
    def publish_state(self, camera_name: str, state: str):
        """
        Publish camera detection state
        
        Args:
            camera_name: Camera name
            state: State ("ON" or "OFF")
        """
        topic = f"{config.MQTT_STATE_TOPIC_PREFIX}/{camera_name}/detection"
        print(f"publish_state {camera_name} {state} {topic}")

        self.mqtt_client.publish(topic, state, retain=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"{timestamp} - [{camera_name}] 📡 MQTT → {state}")
    
    def is_detection_enabled(self, camera_name: str) -> bool:
        """
        Return detection enable state for a specific camera.

        Args:
            camera_name (str): name of the camera, e.g. "kitchen"

        Returns:
            bool: True if detection is enabled, False otherwise
        """
        # print(f"is_detection_enabled {camera_name} {self.detection_enabled.get(camera_name, False)}")
        return self.detection_enabled.get(camera_name, False)
    
    def disconnect(self):
        """Disconnect cleanly from MQTT broker"""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        print("✅ MQTT disconnected from broker")