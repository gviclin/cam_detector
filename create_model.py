from ultralytics import YOLO

# YOLOv8 nano ou YOLO-NAS small
model = YOLO("yolov8n.pt")  # ou "yolonas_s.pt"

# Export en ONNX pour OpenVINO
model.export(format="onnx", opset=17, dynamic=False)
