"""
Centralized configuration for video detection system
"""
import os

# YOLO model parameters
MODEL_PATH = "yolov8n_openvino_model/"
CONF_THRESHOLD = 0.4
CLASS_FILTER = [0]  # 0 = person

# COCO classes (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Camera configuration
CAMERAS = [
    {"name": "kitchen", "url": "rtsp://DY9bKwcB:l9lSQDtDCvLmNZUT@192.168.68.52/live/ch1"},
    {"name": "ext_north", "url": "rtsp://FlZgCcmK:eALNj6HmzLZ9MDzF@192.168.68.54/live/ch1"},
    # {"name": "ext_south", "url": "rtsp://T7iby8nk:jx1IXcoYaUe5jrHs@192.168.68.53/live/ch1"}
]

# Video processing parameters
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 640
VIDEO_FPS = 2
PROCESSING_INTERVAL = 0.5  # seconds between each detection
TARGET_RATIO = 16/9  # Crop ratio

# Output parameters
SAVE_DIR = "output_images"
OFF_DELAY = 4  # delay before switching to OFF after last detection

# MQTT configuration
MQTT_BROKER = "192.168.68.55"
MQTT_PORT = 1883
MQTT_USERNAME = "z2mqtt"
MQTT_PASSWORD = "Viclin4*"
MQTT_ENABLE_TOPIC = "home/camera/detection/enable"
MQTT_STATE_TOPIC_PREFIX = "home/camera"

# RTSP streaming configuration (FFmpeg)
RTSP_OUTPUT_URL = "rtsp://localhost:8554/yolo"
FFMPEG_PRESET = "ultrafast"
FFMPEG_TUNE = "zerolatency"