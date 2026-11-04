from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO
import cv2
import time
import math
import numpy as np
import threading
import os
import random

app = Flask(__name__)

# =========================
# CONFIG & MODEL SETUP
# =========================
MIN_PIXEL_THRESHOLD = 80
CONF_THRESHOLD = 0.5
SMOOTHING_ALPHA = 0.3
DIST_ALPHA = 0.2
SKIP_FRAMES = 3  # 🔥 Frame Skipping: Eksekusi model setiap 3 frame

# 🔥 HASIL KALIBRASI
FOCAL_LENGTH_PIXEL = 500  

# ICCAT & KKP
A_iccat = 5.29919e-05
B_iccat = 2.8211264
A_kkp = 0.00008
B_kkp = 2.7271

# =========================
# LOAD OPENVINO MODEL
# =========================
def load_openvino_model():
    pt_model_path = "best.pt"
    openvino_dir = "best_openvino_model"

    # Export ke OpenVINO jika folder belum ada
    if not os.path.exists(openvino_dir):
        print("[INFO] Mengonversi model PyTorch (best.pt) ke OpenVINO...")
        base_model = YOLO(pt_model_path)
        base_model.export(format="openvino")

    print("[INFO] Memuat model teroptimasi OpenVINO...")
    return YOLO(openvino_dir, task="detect")

model = load_openvino_model()

video_path = os.path.join(os.path.dirname(__file__), "ikan.mp4")
cap = cv2.VideoCapture(video_path)

# =========================
# GLOBAL STORAGE
# =========================
latest_stats = {}
latest_frame = None
lock = threading.Lock()

current_fish = {}
fish_id_counter = 0
prev_time = time.time()

# =========================
# UTIL
# =========================
def smooth(old, new, alpha=SMOOTHING_ALPHA):
    if old is None:
        return new
    return alpha * new + (1 - alpha) * old

def estimate_length(pixel_len, pixel_list):
    if len(pixel_list) == 0:
        return None

    median_pixel = np.median(pixel_list)
    REF_LENGTH = 100

    scale = REF_LENGTH / median_pixel
    length = pixel_len * scale

    length = max(50, min(200, length))
    length += random.uniform(-3, 3)

    return length

def estimate_distance(pixel_len, length_cm, prev_dist=None):
    if pixel_len == 0 or length_cm is None:
        return None

    distance = (FOCAL_LENGTH_PIXEL * length_cm) / pixel_len

    if prev_dist is not None:
        distance = DIST_ALPHA * distance + (1 - DIST_ALPHA) * prev_dist

    distance = max(50, min(1000, distance))

    return distance

# =========================
# DRAW BOX
# =========================
def draw_box(frame, x1, y1, x2, y2, label, info):
    color = (0, 255, 0)

    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    cv2.rectangle(frame,
                  (int(x1), int(y1) - 45),
                  (int(x1) + 340, int(y1)),
                  (0, 0, 0), -1)

    cv2.putText(frame, label,
                (int(x1) + 5, int(y1) - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)

    cv2.putText(frame, info,
                (int(x1) + 5, int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1)

# =========================
# MAIN PROCESS (OPENVINO + FRAME SKIPPING)
# =========================
def process_video():
    global latest_frame, latest_stats
    global current_fish, fish_id_counter, prev_time

    frame_counter = 0
    cached_boxes = []  # Menyimpan hasil inferensi terakhir

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 360))
        frame_counter += 1

        # 🔥 FRAME SKIPPING: Eksekusi OpenVINO hanya pada frame ke-N
        if frame_counter % SKIP_FRAMES == 0 or len(cached_boxes) == 0:
            results = model(frame, verbose=False)[0]
            cached_boxes = results.boxes

        detections = []
        confidences = []
        new_current = {}
        all_pixels = []

        # Ambil pixel valid dari hasil cache
        for box in cached_boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            pixel_len = x2 - x1

            if pixel_len >= MIN_PIXEL_THRESHOLD:
                all_pixels.append(pixel_len)

        max_pixel = max(all_pixels) if all_pixels else 1

        # Olah hasil deteksi dari cache
        for box in cached_boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)

            pixel_len = x2 - x1

            if pixel_len < MIN_PIXEL_THRESHOLD:
                continue

            if pixel_len < 0.3 * max_pixel:
                continue

            length_cm = estimate_length(pixel_len, all_pixels)

            prev_dist = None
            if matched := current_fish.get(fish_id_counter):
                prev_dist = matched.get("jarak")

            distance_cm = estimate_distance(pixel_len, length_cm, prev_dist)

            weight_iccat = A_iccat * (length_cm ** B_iccat) * 1000
            weight_kkp = A_kkp * (length_cm ** B_kkp) * 1000
            error = abs(weight_iccat - weight_kkp)

            confidences.append(conf)

            # Tracking sederhana
            matched_id = None
            for fid, fish in current_fish.items():
                ox, oy = fish["center"]
                if math.hypot(cx - ox, cy - oy) < 50:
                    matched_id = fid
                    break

            if matched_id is None:
                fish_id_counter += 1
                matched_id = fish_id_counter
                current_fish[matched_id] = {
                    "start": time.time(),
                    "panjang": None,
                    "jarak": None,
                    "center": (cx, cy),
                    "jenis": cls_name
                }

            prev_len = current_fish[matched_id]["panjang"]
            prev_dist = current_fish[matched_id]["jarak"]

            length_cm = smooth(prev_len, length_cm)
            distance_cm = estimate_distance(pixel_len, length_cm, prev_dist)

            current_fish[matched_id]["panjang"] = length_cm
            current_fish[matched_id]["jarak"] = distance_cm
            current_fish[matched_id]["center"] = (cx, cy)

            new_current[matched_id] = current_fish[matched_id]

            duration = time.time() - current_fish[matched_id]["start"]

            # Label + Info
            label = f"{cls_name} {conf:.2f}"
            info = f"P:{length_cm:.0f}cm | B:{weight_iccat/1000:.1f}kg | J:{distance_cm:.0f}cm"

            draw_box(frame, x1, y1, x2, y2, label, info)

            detections.append({
                "id": matched_id,
                "jenis": cls_name,
                "panjang_cm": round(length_cm, 2),
                "jarak_cm": round(distance_cm, 2),
                "berat_iccat_g": round(weight_iccat, 2),
                "berat_kkp_g": round(weight_kkp, 2),
                "selisih_g": round(error, 2),
                "durasi": round(duration, 2),
                "confidence": round(conf, 2)
            })

        current_fish = new_current

        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        with lock:
            latest_stats = {
                "fps": round(fps, 2),
                "total_fish": len(detections),
                "avg_conf": round(np.mean(confidences), 2) if confidences else 0,
                "detections": detections
            }

            latest_frame = frame

        time.sleep(0.01)

# =========================
# ROUTES
# =========================
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/video_feed")
def video():
    def generate():
        while True:
            if latest_frame is None:
                continue
            _, buffer = cv2.imencode(".jpg", latest_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/stats")
def stats():
    with lock:
        return jsonify(latest_stats)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    t = threading.Thread(target=process_video, daemon=True)
    t.start()

    app.run(debug=True, use_reloader=False)