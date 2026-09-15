"""
Ring Stack Counter - Web App
=============================
Flask server that:
  1. Serves a mobile-friendly HTML page with a camera-capture button.
  2. Receives the uploaded photo.
  3. Runs the ring-counting algorithm (ported from counter.py).
  4. Returns the ring count + an annotated image.

Run locally:
    pip install -r requirements.txt
    python app.py
    -> open http://127.0.0.1:5000 in a browser

Deploy (get a public URL): see README_deploy.md
"""

import os
import uuid

import cv2
import numpy as np
from scipy.signal import find_peaks
from flask import Flask, request, render_template, jsonify, send_from_directory

app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
class Config:
    MAX_WIDTH = 1000
    CLAHE_CLIP = 3.0
    CLAHE_GRID = (8, 8)
    LOCAL_NORM_WINDOW = 41
    PEAK_DISTANCE_FRAC = 0.010
    PEAK_PROMINENCE_FRAC = 0.08
    GAP_OUTLIER_TOLERANCE = 0.55
    AUTO_DETECT_MIN_AREA_FRAC = 0.05
    AUTO_DETECT_MAX_AREA_FRAC = 0.85
    STRIP_WIDTH_FRAC = 0.12
    BLUR_WARN_THRESHOLD = 80
    BRIGHTNESS_WARN_LOW = 40
    BRIGHTNESS_WARN_HIGH = 220


# --------------------------------------------------------------------------
# ALGORITHM (same as counter.py, auto-detect box, no YOLO/manual needed
# server-side; the browser sends a plain photo)
# --------------------------------------------------------------------------
def resize_image(img, max_width=Config.MAX_WIDTH):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    return cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def enhance_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP, tileGridSize=Config.CLAHE_GRID)
    return clahe.apply(gray)


def check_blur(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_brightness(gray):
    return float(np.mean(gray))


def check_glare(gray, box, overexposed_thresh=240):
    x, y, w, h = box
    region = gray[y:y + h, x:x + w]
    return float(np.mean(region > overexposed_thresh)) if region.size else 0.0


def detect_stack_auto(gray):
    h, w = gray.shape[:2]
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(thresh)

    def border_touch_score(mask):
        border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
        return np.mean(border == 255)

    mask = thresh if border_touch_score(thresh) < border_touch_score(inv) else inv
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, w, h), 1.0, False

    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)
    area_frac = (cw * ch) / float(w * h)
    reliable = Config.AUTO_DETECT_MIN_AREA_FRAC <= area_frac <= Config.AUTO_DETECT_MAX_AREA_FRAC
    if reliable:
        margin_x = int(cw * 0.03)
        margin_y = int(ch * 0.03)
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        cw = min(w - x, cw + 2 * margin_x)
        ch = min(h - y, ch + 2 * margin_y)
    return (x, y, cw, ch), area_frac, reliable


def extract_profile(gray, box):
    x, y, w, h = box
    strip_w = max(3, int(w * Config.STRIP_WIDTH_FRAC))
    cx = x + w // 2
    x0 = max(x, cx - strip_w // 2)
    x1 = min(x + w, x0 + strip_w)
    strip = gray[y:y + h, x0:x1]
    profile = np.mean(strip.astype(np.float64), axis=1)
    return profile, (x0, y, x1 - x0, h)


def local_contrast_normalize(signal, window):
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(signal, pad, mode="reflect")
    kernel = np.ones(window) / window
    local_mean = np.convolve(padded, kernel, mode="valid")
    local_sq_mean = np.convolve(padded ** 2, kernel, mode="valid")
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 1e-6))
    return (signal - local_mean) / local_std


def compute_gradient(profile):
    return np.abs(np.gradient(profile))


def find_ring_peaks(gradient):
    length = len(gradient)
    if length < 5:
        return np.array([], dtype=int)
    rng = gradient.max() - gradient.min()
    prominence = max(rng * Config.PEAK_PROMINENCE_FRAC, 1e-6)
    distance = max(1, int(length * Config.PEAK_DISTANCE_FRAC))
    peaks, _ = find_peaks(gradient, prominence=prominence, distance=distance)
    return peaks


def validate_gaps(peaks):
    if len(peaks) < 3:
        return peaks, 0.0
    gaps = np.diff(peaks)
    median_gap = np.median(gaps)
    if median_gap <= 0:
        return peaks, 0.0
    keep = [True] * len(peaks)
    for i, gap in enumerate(gaps):
        if abs(gap - median_gap) / median_gap > Config.GAP_OUTLIER_TOLERANCE:
            keep[i + 1] = False
    filtered = peaks[np.array(keep)]
    gaps_after = np.diff(filtered) if len(filtered) > 1 else np.array([median_gap])
    consistency = 1.0 - min(1.0, float(np.std(gaps_after) / (np.mean(gaps_after) + 1e-9)))
    return filtered, max(0.0, consistency)


def count_rings(filtered_peaks):
    return max(0, len(filtered_peaks) - 1) if len(filtered_peaks) >= 2 else len(filtered_peaks)


def compute_confidence(consistency, blur_score, brightness, n_peaks, box_reliable):
    score = consistency
    if not box_reliable:
        score -= 0.5
    if blur_score < Config.BLUR_WARN_THRESHOLD:
        score -= 0.25
    if brightness < Config.BRIGHTNESS_WARN_LOW or brightness > Config.BRIGHTNESS_WARN_HIGH:
        score -= 0.15
    if n_peaks < 4:
        score -= 0.2
    if score >= 0.6:
        return "High"
    elif score >= 0.3:
        return "Medium"
    return "Low"


def process_image(path, out_path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read uploaded image")
    img = resize_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_score = check_blur(gray)
    brightness = check_brightness(gray)
    enhanced = enhance_contrast(gray)

    stack_box, area_frac, box_reliable = detect_stack_auto(enhanced)
    glare = check_glare(gray, stack_box)

    profile, strip_box = extract_profile(enhanced, stack_box)
    normalized = local_contrast_normalize(profile, Config.LOCAL_NORM_WINDOW)
    gradient = compute_gradient(normalized)
    peaks = find_ring_peaks(gradient)
    filtered_peaks, consistency = validate_gaps(peaks)
    ring_count = count_rings(filtered_peaks)
    confidence = compute_confidence(consistency, blur_score, brightness, len(filtered_peaks), box_reliable)

    annotated = img.copy()
    sx, sy, sw, sh = stack_box
    rx, ry, rw, rh = strip_box
    cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), (0, 165, 255), 2)
    cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 1)
    for p in filtered_peaks:
        yy = ry + int(p)
        cv2.line(annotated, (sx, yy), (sx + sw, yy), (0, 255, 0), 1)
    cv2.putText(annotated, str(ring_count), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(out_path, annotated)

    return {
        "ring_count": ring_count,
        "confidence": confidence,
        "consistency": round(consistency, 2),
        "blur_score": round(blur_score, 1),
        "brightness": round(brightness, 1),
        "glare": round(glare * 100, 1),
        "box_reliable": box_reliable,
    }


# --------------------------------------------------------------------------
# ROUTES
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/count", methods=["POST"])
def count():
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    file = request.files["photo"]
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    uid = uuid.uuid4().hex
    in_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
    out_path = os.path.join(OUTPUT_DIR, f"{uid}_out.jpg")
    file.save(in_path)

    try:
        result = process_image(in_path, out_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    result["image_url"] = f"/result-image/{uid}_out.jpg"
    return jsonify(result)


@app.route("/result-image/<filename>")
def result_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
