"""
Ring Stack Counter - Web App  (production build)
=================================================
Flask server:
  1. Serves a mobile-friendly HTML page.
  2. Receives photo (multipart/form-data).
  3. Runs ring-counting algorithm.
  4. Returns JSON with count + base64 annotated image.

Local run:
    pip install -r requirements.txt
    python app.py
    -> http://127.0.0.1:5000

Deploy: push to GitHub, connect to Vercel (see README_deploy.md)
"""

import os
import base64
import tempfile
import logging

import cv2
import numpy as np
from scipy.signal import find_peaks
from flask import Flask, request, render_template, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB hard cap

UPLOAD_DIR = tempfile.gettempdir()


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# IMAGE PROCESSING HELPERS
# ─────────────────────────────────────────────
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


def check_glare(gray, box, thresh=240):
    x, y, w, h = box
    region = gray[y:y + h, x:x + w]
    return float(np.mean(region > thresh)) if region.size else 0.0


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
        mx = int(cw * 0.03)
        my = int(ch * 0.03)
        x = max(0, x - mx)
        y = max(0, y - my)
        cw = min(w - x, cw + 2 * mx)
        ch = min(h - y, ch + 2 * my)
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
    window = max(3, window | 1)
    pad = window // 2
    padded = np.pad(signal, pad, mode="reflect")
    k = np.ones(window) / window
    lm = np.convolve(padded, k, mode="valid")
    lsm = np.convolve(padded ** 2, k, mode="valid")
    lstd = np.sqrt(np.maximum(lsm - lm ** 2, 1e-6))
    return (signal - lm) / lstd


def find_ring_peaks(gradient):
    n = len(gradient)
    if n < 5:
        return np.array([], dtype=int)
    rng = gradient.max() - gradient.min()
    prominence = max(rng * Config.PEAK_PROMINENCE_FRAC, 1e-6)
    distance = max(1, int(n * Config.PEAK_DISTANCE_FRAC))
    peaks, _ = find_peaks(gradient, prominence=prominence, distance=distance)
    return peaks


def validate_gaps(peaks):
    if len(peaks) < 3:
        return peaks, 0.0
    gaps = np.diff(peaks)
    med = np.median(gaps)
    if med <= 0:
        return peaks, 0.0
    keep = [True] + [abs(g - med) / med <= Config.GAP_OUTLIER_TOLERANCE for g in gaps]
    filtered = peaks[np.array(keep)]
    g2 = np.diff(filtered) if len(filtered) > 1 else np.array([med])
    consistency = 1.0 - min(1.0, float(np.std(g2) / (np.mean(g2) + 1e-9)))
    return filtered, max(0.0, consistency)


def count_rings(filtered_peaks):
    n = len(filtered_peaks)
    return max(0, n - 1) if n >= 2 else n


def compute_confidence(consistency, blur, brightness, n_peaks, box_reliable):
    score = consistency
    if not box_reliable:  score -= 0.5
    if blur < Config.BLUR_WARN_THRESHOLD: score -= 0.25
    if brightness < Config.BRIGHTNESS_WARN_LOW or brightness > Config.BRIGHTNESS_WARN_HIGH:
        score -= 0.15
    if n_peaks < 4: score -= 0.2
    if score >= 0.6: return "High"
    if score >= 0.3: return "Medium"
    return "Low"


def process_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read image. Please upload a valid JPG/PNG.")

    img = resize_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_score = check_blur(gray)
    brightness = check_brightness(gray)
    enhanced = enhance_contrast(gray)

    stack_box, area_frac, box_reliable = detect_stack_auto(enhanced)
    glare = check_glare(gray, stack_box)

    profile, strip_box = extract_profile(enhanced, stack_box)
    normalized = local_contrast_normalize(profile, Config.LOCAL_NORM_WINDOW)
    gradient = np.abs(np.gradient(normalized))
    peaks = find_ring_peaks(gradient)
    filtered_peaks, consistency = validate_gaps(peaks)
    ring_count = count_rings(filtered_peaks)
    confidence = compute_confidence(consistency, blur_score, brightness,
                                    len(filtered_peaks), box_reliable)

    # ── Annotate ──
    annotated = img.copy()
    sx, sy, sw, sh = stack_box
    rx, ry, rw, rh = strip_box

    # Stack bounding box (orange)
    cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), (0, 165, 255), 2)
    # Analysis strip (blue)
    cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (255, 80, 0), 1)
    # Ring lines (green)
    for p in filtered_peaks:
        yy = ry + int(p)
        cv2.line(annotated, (sx, yy), (sx + sw, yy), (0, 255, 80), 1)
    # Count overlay
    cv2.putText(annotated, f"{ring_count}", (12, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(annotated, f"{ring_count}", (12, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 255), 3, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("Could not encode result image")

    image_data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")
    logger.info(f"Result: {ring_count} rings | confidence={confidence} | blur={blur_score:.1f}")

    return {
        "ring_count": ring_count,
        "confidence": confidence,
        "consistency": round(float(consistency), 2),
        "blur_score": round(blur_score, 1),
        "brightness": round(brightness, 1),
        "glare": round(glare * 100, 1),
        "box_reliable": box_reliable,
        "image_data_url": image_data_url,
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/count", methods=["POST"])
def count():
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    file = request.files["photo"]
    if file.filename == "":
        return jsonify({"error": "Empty file received"}), 400

    ext = os.path.splitext(file.filename or "photo.jpg")[1].lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        return jsonify({"error": "Unsupported file type. Use JPG or PNG."}), 400

    in_path = os.path.join(UPLOAD_DIR, f"ring_upload_{os.getpid()}{ext}")
    try:
        file.save(in_path)
        result = process_image(in_path)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Image too large. Please use a photo under 10 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
