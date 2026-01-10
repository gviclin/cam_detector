from openvino.tools.pot import IEEngine, load_model, save_model
from openvino.tools.pot import DataLoader, create_pipeline

# Dataset calibration
class MyDataLoader(DataLoader):
    def __init__(self, image_folder):
        import glob
        self.files = glob.glob(image_folder + "/*.jpg")
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        import cv2, numpy as np
        img = cv2.imread(self.files[idx])
        img = cv2.resize(img, (640,640))
        img = img.transpose(2,0,1) / 255.0
        img = img[np.newaxis, :, :, :].astype("float32")
        return (img, None)

calib_loader = MyDataLoader("calib_dataset")  # dossier avec 50–500 images représentatives

# Charger modèle
model_config = {"model_name": "yolov8n", "model": "yolov8n.onnx"}
model = load_model(model_config)

# Engine
engine = IEEngine(config={}, data_loader=calib_loader, metric=None)

# Pipeline INT8
from openvino.tools.pot.algorithms.quantization.default import DefaultQuantization
algorithms = [{"name": "DefaultQuantization", "params": {"target_device": "CPU", "preset": "performance"}}]

pipeline = create_pipeline(algorithms, engine)
compressed_model = pipeline.run(model)

# Sauvegarder modèle INT8
save_model(compressed_model, save_path="yolov8n_int8")
