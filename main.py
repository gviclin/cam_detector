import cv2, os, time, sys
import numpy as np
from openvino import Core

RTSP_URLS = [
    "rtsp://DY9bKwcB:l9lSQDtDCvLmNZUT@192.168.68.52/live/ch1", # kitchen
    # "rtsp://FlZgCcmK:eALNj6HmzLZ9MDzF@192.168.68.54/live/ch1", # ext nord
    # "rtsp://T7iby8nk:jx1IXcoYaUe5jrHs@192.168.68.53/live/ch1", # ext sud
]
MODEL_PATH = "yolov8n.onnx"  # modèle ONNX
SAVE_DIR = "output_images"        # répertoire de sauvegarde
# MODEL_PATH = "yolov8n_int8/optimized.xml"  # OpenVINO INT8
CONF_THRESHOLD = 0.4
CLASS_FILTER = [0]                # classes à détecter
FRAME_SKIP = 10
ALERT_COOLDOWN = 10  # secondes

# OpenVINO
ie = Core()
# Charger le modèle ONNX
compiled_model = ie.compile_model(model=MODEL_PATH, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

os.makedirs(SAVE_DIR, exist_ok=True)
for f in os.listdir(SAVE_DIR):
    file_path = os.path.join(SAVE_DIR, f)
    if os.path.isfile(file_path):
        os.remove(file_path)
print(f"[INFO] Cleared directory: {SAVE_DIR}")

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

def preprocess(frame):
    frame = crop_sides(frame)
    framePreprocessed = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(framePreprocessed, cv2.COLOR_BGR2RGB)
    img = img.transpose(2,0,1) / 255.0

    return framePreprocessed, np.expand_dims(img, axis=0).astype(np.float32)

# Post-traitement YOLO
def postprocess(output, conf_threshold=0.4, class_filter=[0]):
    detections = []
    for pred in output[0]:
        conf = float(pred[4])
        class_id = int(np.argmax(pred[5:]))
        class_conf = float(pred[5 + class_id])
        score = conf * class_conf
        if class_id < 10000:
            print(f"detected class_id {class_id}, score {score}")
        if score < conf_threshold:
            continue
        if class_id not in class_filter:
            continue
        x, y, w, h = pred[:4]
        x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
        detections.append([x1,y1,x2,y2,score])

        # Dessiner la bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    return detections

# Détection flux RTSP
def detect_rtsp(url):

    img = cv2.imread("person.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # normaliser 0-1
    img = cv2.resize(img, (640, 640))     # taille du modèle
    img = np.transpose(img, (2,0,1))      # HWC -> CHW
    img = np.expand_dims(img, 0)          # ajouter batch


    # save_path = os.path.join(SAVE_DIR, f"frame_resize.jpg")
    # cv2.imwrite(save_path, frame_resized)

    output = compiled_model([img])[output_layer]
    detections = postprocess(output, CONF_THRESHOLD, CLASS_FILTER)



    sys.exit(1)


    print("[INFO] Init video capture")
    cap = cv2.VideoCapture(url)
    frame_id = 0
    last_alert = 0

    print("[INFO] Begin video capture")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print(f"⚠ Flux interrompu {url}, reconnexion...")
            time.sleep(2)
            cap = cv2.VideoCapture(url)
            continue

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue


        frame_resized, input_tensor = preprocess(frame)
        output = compiled_model([input_tensor])[output_layer]
        detections = postprocess(output, CONF_THRESHOLD, CLASS_FILTER)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_path = os.path.join(SAVE_DIR, f"{timestamp} {frame_id:04d}.jpg")
        cv2.imwrite(save_path, frame_resized)

        # print(f"type of framePreprocessed{type(framePreprocessed)}")

        # if frame_id > 80:
        #     break

        if detections:
            now = time.time()
            if now - last_alert > ALERT_COOLDOWN:
                print(f"🚨 {now} [cam {url}] {len(detections)} personne(s) détectée(s)")
                last_alert = now

# Multi-caméras (threads)
from threading import Thread
threads = []
for url in RTSP_URLS:
    t = Thread(target=detect_rtsp, args=(url,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
