"""
MQTT manager for home automation system communication
"""
import paho.mqtt.client as mqtt
from datetime import datetime
from typing import Callable, Optional
import config


class MQTTManager:
    """MQTT communication manager"""
    
    def __init__(self, on_enable_change: Optional[Callable[[bool], None]] = None):
        """
        Initialize MQTT manager
        
        Args:
            on_enable_change: Callback called when enable state changes
        """
        self.client = mqtt.Client(client_id="cam_detector")
        self.client.username_pw_set(config.MQTT_USERNAME, config.MQTT_PASSWORD)
        self.client.on_message = self._on_message
        self.on_enable_change = on_enable_change
        self.detection_enabled = True
        
    def _on_message(self, client, userdata, msg):
        """Callback for received MQTT messages"""
        payload = msg.payload.decode()
        
        if msg.topic == config.MQTT_ENABLE_TOPIC:
            if payload == "ON":
                self.detection_enabled = True
                print("🟢 Detection ENABLED")
            elif payload == "OFF":
                self.detection_enabled = False
                print("🔴 Detection DISABLED")
            
            if self.on_enable_change:
                self.on_enable_change(self.detection_enabled)
    
    def connect(self):
        """Connect to MQTT broker"""
        self.client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
        self.client.subscribe(config.MQTT_ENABLE_TOPIC)
        self.client.loop_start()
        print(f"✅ Connected to MQTT broker {config.MQTT_BROKER}:{config.MQTT_PORT}")
    
    def publish_state(self, camera_name: str, state: str):
        """
        Publish camera detection state
        
        Args:
            camera_name: Camera name
            state: State ("ON" or "OFF")
        """
        topic = f"{config.MQTT_STATE_TOPIC_PREFIX}/{camera_name}/detection"
        self.client.publish(topic, state, retain=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"{timestamp} - [{camera_name}] 📡 MQTT → {state}")
    
    def is_detection_enabled(self) -> bool:
        """Return detection enable state"""
        return self.detection_enabled
    
    def disconnect(self):
        """Disconnect cleanly from MQTT broker"""
        self.client.loop_stop()
        self.client.disconnect()
        print("❌ Disconnected from MQTT broker")