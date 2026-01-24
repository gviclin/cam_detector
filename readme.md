stty sane
stty echo


wget https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_linux_amd64.tar.gz
tar -xzf mediamtx_linux_amd64.tar.gz

./mediamtx
rtsp://192.168.68.64:8554/

test mqtt :
mosquitto_sub -h 192.168.68.55 -p 1883 -u z2mqtt -P Viclin4* -t home/camera/# -v


# 🎥 Multi-Camera Surveillance System with AI Detection

Real-time video surveillance system using YOLO (OpenVINO) for person detection, with MQTT integration for home automation.

## 📁 Project Structure

```
.
├── config.py              # Centralized configuration
├── mqtt_manager.py        # MQTT communication management
├── video_processor.py     # Video stream processing
├── detector.py            # YOLO detection
├── rtsp_streamer.py       # RTSP streaming (FFmpeg)
├── camera_monitor.py      # Camera surveillance
├── main.py               # Main application
├── requirements.txt      # Python dependencies
└── output_images/        # Images with detections (generated)
```

## 🚀 Installation

### 1. Prerequisites

- Python 3.8+
- FFmpeg (for optional RTSP streaming)
- OpenVINO Runtime (for Intel acceleration)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configuration

Edit `config.py` to configure:

- **Cameras**: RTSP URLs and names
- **MQTT**: Broker, credentials, topics
- **Detection**: Confidence thresholds, target classes
- **Video**: Resolution, FPS, processing intervals

## 🎯 Usage

### Launch system

```bash
cd ~/sandbox/cam_detector
source venv/bin/activate
python main.py
```

### MQTT commands

**Enable/Disable detection**:
```bash
# Enable
mosquitto_pub -h 192.168.68.55 -u z2mqtt -P "Viclin4*" \
  -t "home/camera/detection/enable" -m "ON"

# Disable
mosquitto_pub -h 192.168.68.55 -u z2mqtt -P "Viclin4*" \
  -t "home/camera/detection/enable" -m "OFF"
```

### MQTT topics

- **Command**: `home/camera/detection/enable` (ON/OFF)
- **States**: `home/camera/{camera_name}/detection` (ON/OFF)

## 🔧 Module Architecture

### `config.py`
Centralized configuration: cameras, MQTT, YOLO, video.

### `mqtt_manager.py`
- MQTT connection management
- Detection state publishing
- Enable command subscription

### `video_processor.py`
- `VideoProcessor`: Frame preprocessing (crop, resize)
- `RTSPCapture`: RTSP capture with auto-reconnection

### `detector.py`
- Object detection with YOLO + OpenVINO
- Filtering by class and confidence
- Structured `Detection` objects

### `rtsp_streamer.py`
RTSP streaming via FFmpeg (optional, currently unused).

### `camera_monitor.py`
- Camera surveillance
- Periodic frame processing
- Detection saving
- ON/OFF state management with delay

### `main.py`
- Multi-threaded orchestration
- System lifecycle management

## 📊 Features

- ✅ Real-time person detection (YOLO)
- ✅ Multi-camera parallel processing (threads)
- ✅ MQTT integration for home automation
- ✅ Automatic detection saving
- ✅ Automatic RTSP stream reconnection
- ✅ Intelligent throttling (avoid CPU overload)
- ✅ ON/OFF states with configurable delay
- ✅ OpenVINO support (Intel CPU/GPU/NPU)

## ⚙️ Customization

### Change detected class

In `camera_monitor.py`, replace `detect_persons()` with:

```python
# Detect multiple classes
detections = self.detector.detect(
    processed_frame, 
    target_classes=["person", "car", "dog"]
)
```

### Change compute device

In `config.py` or during initialization:

```python
detector = YOLODetector(device="intel:gpu")  # GPU
detector = YOLODetector(device="intel:npu")  # NPU
```

### Add RTSP streaming

In `camera_monitor.py`, add:

```python
from rtsp_streamer import RTSPStreamer

# In __init__
self.streamer = RTSPStreamer()
self.streamer.start()

# In process_frame (after detection)
self.streamer.write_frame(processed_frame)
```

## 📝 Notes

- Images are **not saved** for "kitchen" camera
- 3s delay before switching to OFF after last detection
- Processing interval: 1 frame/second
- Confidence threshold: 40%
- Home assistant configuration :
```yaml
binary_sensor:
  - name: "camera_kitchen_detection"
    state_topic: "home/camera/kitchen/detection"
    payload_on: "ON"
    payload_off: "OFF"
    device_class: motion

  - name: "camera_ext_north_detection"
    state_topic: "home/camera/ext_north/detection"
    payload_on: "ON"
    payload_off: "OFF"
    device_class: motion

  - name: "camera_ext_south_detection"
    state_topic: "home/camera/ext_south/detection"
    payload_on: "ON"
    payload_off: "OFF"
    device_class: motion

switch:
  - name: "camera_detector_kitchen_switch"
    command_topic: "home/camera/kitchen/detection/enable"
    state_topic: "home/camera/kitchen/detection/enable"
    payload_on: "ON"
    payload_off: "OFF"
    optimistic: false

  - name: "camera_detector_ext_north_switch"
    command_topic: "home/camera/ext_north/detection/enable"
    state_topic: "home/camera/ext_north/detection/enable"
    payload_on: "ON"
    payload_off: "OFF"
    optimistic: false

  - name: "camera_detector_ext_south_switch"
    command_topic: "home/camera/ext_south/detection/enable"
    state_topic: "home/camera/ext_south/detection/enable"
    payload_on: "ON"
    payload_off: "OFF"
    optimistic: false
```

## 🐛 Troubleshooting

**RTSP connection error**:
- Check camera URLs and credentials
- Test with VLC: `vlc rtsp://...`

**Slow detection**:
- Reduce resolution in `config.py`
- Increase `PROCESSING_INTERVAL`
- Use GPU/NPU if available

**MQTT not responding**:
- Check broker: `mosquitto_sub -h IP -v -t '#'`
- Verify credentials and topics

## 📄 License

Personal project - Free to use