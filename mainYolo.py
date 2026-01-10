import cv2, os, time, sys
from datetime import datetime
import numpy as np
import subprocess
from ultralytics import YOLO

# COCO 80 classes
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
    "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
    "pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv",
    "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]


RTSP_URLS = [
    "rtsp://DY9bKwcB:l9lSQDtDCvLmNZUT@192.168.68.52/live/ch1", # kitchen
    # "rtsp://FlZgCcmK:eALNj6HmzLZ9MDzF@192.168.68.54/live/ch1", # ext nord
    # "rtsp://T7iby8nk:jx1IXcoYaUe5jrHs@192.168.68.53/live/ch1", # ext sud
]
MODEL_PATH = "yolov8n.onnx"  # modèle ONNX
SAVE_DIR = "output_images"        # répertoire de sauvegarde
# MODEL_PATH = "yolov8n_int8/optimized.xml"  # OpenVINO INT8
CONF_THRESHOLD = 0.3
CLASS_FILTER = [0]                # classes à détecter
FRAME_SKIP = 7
ALERT_COOLDOWN = 10  # secondes

# # Load a YOLO11n PyTorch model
# model = YOLO("yolov8n.pt")

# # Export the model
# model.export(format="openvino")  # creates 'yolo11n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO("yolov8n_openvino_model/")

# export DIR
os.makedirs(SAVE_DIR, exist_ok=True)
for f in os.listdir(SAVE_DIR):
    file_path = os.path.join(SAVE_DIR, f)
    if os.path.isfile(file_path):
        os.remove(file_path)
print(f"[INFO] Cleared directory: {SAVE_DIR}")


WIDTH, HEIGHT = 640, 640
FPS = 2

# Lancer FFmpeg en sous-processus
ffmpeg_cmd = [
    "ffmpeg",
    "-loglevel", "warning",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{WIDTH}x{WIDTH}",
    "-r", f"{FPS}",
    "-i", "-",
    "-an",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-g", f"{FPS}",
    "-keyint_min", f"{FPS}",
    "-f", "rtsp",
    "rtsp://localhost:8554/yolo"
]

# Prétraitement
def crop_sides(img, target_ratio=4/3):
    """
    Crop horizontalement pour atteindre un ratio cible.
    - img : image BGR ou RGB (H, W, C)
    - target_ratio : largeur/hauteur souhaité (ex : 1.6 pour légèrement plus étroit que 16/9)
    Retour : image recadrée avec le ratio target_ratio
    """
    h, w = img.shape[:2]
    current_ratio = w / h

    if current_ratio <= target_ratio:
        # Image déjà plus étroite que le ratio cible, pas de crop nécessaire
        return img
    
    # Nouvelle largeur après crop
    new_w = int(h * target_ratio)
    left = (w - new_w) // 2
    cropped = img[:, left:left+new_w]
    return cropped




# Détection flux RTSP
def detect_rtsp(url):

    print("[INFO] Init video capture")
    cap = cv2.VideoCapture(url)
    frame_id = 0
    last_alert = 0

    OutProcess = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    print("[INFO] Begin video capture")

    while True:
        ret, img = cap.read()
        
        if not ret:
            print(f"⚠ Flux interrompu {url}, reconnexion...")
            time.sleep(2)
            cap = cv2.VideoCapture(url)
            continue

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue


        # Charger une image (ou URL)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = crop_sides(img)

        img = cv2.resize(img, (WIDTH, HEIGHT))


        # Run inference with specified device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
        results = ov_model(img, device="intel:cpu")


        # Dessiner les résultats sur l'image
        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                class_id = int(cls)
                class_name = COCO_CLASSES[class_id]
                x1, y1, x2, y2 = map(int, box)
                proba = 100 * conf
                label = f"{class_name} {proba:.0f}%"

                if not "potted plant" in class_name and not "person" in class_name:
                    continue

                if conf < CONF_THRESHOLD:
                    continue            

                # print(f"class_id={class_id}, class_name={class_name}, confidence={conf:.3f}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                
                # Dessiner la bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                # Écrire le label
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)

    
        
        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        # save_path = os.path.join(SAVE_DIR, f"{timestamp}- {frame_id:04d}.jpg")
        # cv2.imwrite(save_path, img)

        # Envoyer la frame à FFmpeg
        OutProcess.stdin.write(img.tobytes())

    cap.release()
    process.stdin.close()
    process.wait()


# Multi-caméras (threads)
from threading import Thread
threads = []
for url in RTSP_URLS:
    t = Thread(target=detect_rtsp, args=(url,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()


