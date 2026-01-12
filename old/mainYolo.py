import cv2, os, time, sys, shutil
from datetime import datetime
import numpy as np
import subprocess
import paho.mqtt.client as mqtt

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


CAMS = [
    {"name": "kitchen", "url": "rtsp://DY9bKwcB:l9lSQDtDCvLmNZUT@192.168.68.52/live/ch1"},
    {"name": "ext_north", "url": "rtsp://FlZgCcmK:eALNj6HmzLZ9MDzF@192.168.68.54/live/ch1"},
    {"name": "ext_south", "url": "rtsp://T7iby8nk:jx1IXcoYaUe5jrHs@192.168.68.53/live/ch1"},
]



MODEL_PATH = "yolov8n.onnx"  # modèle ONNX
SAVE_DIR = "output_images"        # répertoire de sauvegarde
# MODEL_PATH = "yolov8n_int8/optimized.xml"  # OpenVINO INT8
CONF_THRESHOLD = 0.4
CLASS_FILTER = [0]                # classes à détecter

interval = 1  # 1 frame toutes les 0.5s → 2 FPS max
ALERT_COOLDOWN = 10  # secondes

# # Load a YOLO11n PyTorch model
# model = YOLO("yolov8n.pt")

# # Export the model
# model.export(format="openvino")  # creates 'yolo11n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO("yolov8n_openvino_model/")
 
    


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

# MQTT
MQTT_BROKER = "192.168.68.55"
MQTT_PORT = 1883
MQTT_USERNAME = "z2mqtt"              # facultatif
MQTT_PASSWORD = "Viclin4*"              # facultatif

OFF_DELAY = 3
ENABLE_TOPIC = "home/camera/detection/enable"

detect_enabled = True

def on_message(client, userdata, msg):
    global detect_enabled
    payload = msg.payload.decode()
    if payload == "ON":
        detect_enabled = True
        print("🟢 Détection ACTIVÉE")
    elif payload == "OFF":
        detect_enabled = False
        print("🔴 Détection DÉSACTIVÉE")

def publish_state(new_state, cam):
    mqtt_client.publish(f"home/camera/{cam['name']}/detection", new_state, retain=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"{timestamp} - [{cam['name']}] 📡 MQTT → {new_state}")



mqtt_client = mqtt.Client(client_id="cam_detector")
mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.subscribe(ENABLE_TOPIC)
mqtt_client.loop_start()

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
def detect_rtsp(cam):
    cam_name = cam['name']
    print(f"[{cam_name}] Init video capture")
    start = time.time()
    cap = cv2.VideoCapture(cam["url"])

    last_detection_time = 0

    # cap.set(cv2.CAP_PROP_FPS, 5)          # 5 images par seconde
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    end = time.time()
    print(f"[{cam_name}] End init video capture. Duration : {(end - start) * 1000:.2f} ms")

    frame_id = 0
    last_alert = 0

    cam_dir = os.path.join(SAVE_DIR, cam["name"])
    
    # Supprimer le contenu s'il existe
    if os.path.exists(cam_dir):
        shutil.rmtree(cam_dir)
    
    # Recréer le répertoire vide
    os.makedirs(cam_dir, exist_ok=True)
    
    print(f"Répertoire prêt et vidé : {cam_dir}")


    OutProcess = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    print(f"[{cam_name}] Begin video capture")

    # Variables pour calculer le FPS
    prev_time = datetime.now()

    last_processed = 0

    state = "OFF"
    publish_state(state, cam)

    while True:
        ret, img = cap.read()

        # # Calcul diff / trame prec
        # curr_time = datetime.now()
        # dt = (curr_time - prev_time).total_seconds()
        # prev_time = curr_time
        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        # print(f"{timestamp} - [{cam_name}] diff : {dt} ")


        # affiche infos
        # print(f"[{cam_name}] ")

        # Lire les infos
        
        # width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps    = cap.get(cv2.CAP_PROP_FPS)

        # print(f"{timestamp} - [{cam_name}]🔹 Résolution : {width}x{height}  FPS : {fps}")
        
        if not ret:
            print(f"[{cam_name}] Flux stopped {cam['url']}, reconnection...")
            time.sleep(2)
            cap = cv2.VideoCapture(cam["url"])
            continue

        if not detect_enabled:
            # Si désactivé → forcer OFF une seule fois
            if state != "OFF":
                publish_state("ON", cam)
            continue
        
        frame_id += 1
        now = time.time()
        if now - last_processed >= interval:
            last_processed = now

            # Charger une image (ou URL)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = crop_sides(img)

            img = cv2.resize(img, (WIDTH, HEIGHT))


            # Run inference with specified device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
            start = time.time()
            results = ov_model(img, device="intel:cpu", verbose=False)
            end = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            # print(f"{timestamp} - [{cam_name}] Inference Duration : {(end - start) * 1000:.2f} ms")

            detected = False
            # Dessiner les résultats sur l'image
            for r in results:            
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    class_id = int(cls)
                    class_name = COCO_CLASSES[class_id]
                    x1, y1, x2, y2 = map(int, box)
                    proba = 100 * conf
                    label = f"{class_name} {proba:.0f}%"

                    # if not "bicycle" in class_name and not "person" in class_name and not "cat" in class_name and not "dog" in class_name and not "bird" in class_name:
                    if  not "person" in class_name:
                        continue

                    if conf < CONF_THRESHOLD:
                        continue            
                    
                    detected = True
                    # print(f"[{cam_name}] class_id={class_id}, class_name={class_name}, confidence={conf:.3f}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                    
                    # Dessiner la bbox
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                    # Écrire le label
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 255, 0), 2)


            if detected:
                last_detection_time = now
                if state=="OFF":
                    state = "ON"
                    publish_state(state, cam)
                
                if not "kitchen" in cam["name"]:
                    save_path = os.path.join(SAVE_DIR, f"{timestamp}- {frame_id:04d}.jpg")
                    save_path = os.path.join(cam_dir, f"{timestamp}- {frame_id:04d}.jpg")      
                    cv2.imwrite(save_path, img)

            else:
                if state == "ON" and (now - last_detection_time) > OFF_DELAY:
                    state = "OFF"
                    publish_state(state, cam)
                        


            # # Envoyer la frame à FFmpeg
            # OutProcess.stdin.write(img.tobytes())

    cap.release()
    process.stdin.close()
    process.wait()


# Multi-caméras (threads)
from threading import Thread
threads = []
for cam in CAMS:
    t = Thread(target=detect_rtsp, args=(cam,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()


