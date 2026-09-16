"""
Ring Stack Counter - Web App (v3 - fixed detection)
=====================================================
Key fix vs previous version: detect_stack_auto() (global Otsu threshold)
silently fell back to the FULL FRAME whenever the background was close in
tone to the stack (common on factory-floor photos with weathered metal,
rust, grating). That produced a big confident-looking number that was
actually counting background texture. This version:

  1. Locates the stack via vertical-strip ridge-density scanning instead
     of global thresholding -- far more robust on cluttered scenes.
  2. Refuses to guess: if no column passes a minimum periodicity bar,
     returns an explicit error instead of a silently-wrong count.

Local run:
    pip install -r requirements.txt
    python app.py
    -> http://127.0.0.1:5000

Deploy: push to GitHub, import into Vercel (vercel.json included).
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
    MAX_WIDTH = 1100
    CLAHE_CLIP = 3.0
    CLAHE_GRID = (8, 8)

    # --- stack localization (replaces old Otsu box-detect) ---
    COLUMN_SCAN_STEP_FRAC = 0.02        # scan every 2% of width
    COLUMN_WIDTH_FRAC = 0.16            # width of each candidate column
    MIN_RIDGE_COUNT = 6                 # a real stack has many ring seams
    MIN_PERIODICITY = 0.35              # how regular the spacing must be
    VERTICAL_MARGIN_FRAC = 0.03         # trim a little off top/bottom

    LOCAL_NORM_WINDOW = 41
    PEAK_DISTANCE_FRAC = 0.010
    PEAK_PROMINENCE_FRAC = 0.10
    GAP_OUTLIER_TOLERANCE = 0.50

    BLUR_WARN_THRESHOLD = 80
    BRIGHTNESS_WARN_LOW = 35
    BRIGHTNESS_WARN_HIGH = 225


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


def _column_profile(enhanced, x0, x1, y0, y1):
    """Mean intensity down a vertical strip -> the raw 'ring signal'."""
    strip = enhanced[y0:y1, x0:x1]
    return np.mean(strip.astype(np.float64), axis=1)


def _ridge_score(profile):
    """
    How strongly and regularly this column shows periodic ring seams.
    Real ring stacks produce many evenly-spaced sharp intensity dips;
    background clutter (grating, wall texture) doesn't.
    Returns (n_peaks, periodicity 0..1).
    """
    if len(profile) < 20:
        return 0, 0.0
    grad = np.abs(np.gradient(profile))
    rng = grad.max() - grad.min()
    if rng < 1e-6:
        return 0, 0.0
    prominence = max(rng * 0.12, 1e-6)
    distance = max(1, int(len(profile) * 0.008))
    peaks, _ = find_peaks(grad, prominence=prominence, distance=distance)
    if len(peaks) < 3:
        return len(peaks), 0.0
    gaps = np.diff(peaks)
    med = np.median(gaps)
    if med <= 0:
        return len(peaks), 0.0
    periodicity = 1.0 - min(1.0, float(np.std(gaps) / med))
    return len(peaks), max(0.0, periodicity)


def locate_stack_column(enhanced):
    """
    Scan candidate vertical columns across the image and pick the one
    with the strongest, most regular ridge pattern -- i.e. the actual
    ring stack -- instead of trying to threshold the whole stack shape
    out of a cluttered background.

    Returns (box, reliable) where box = (x, y, w, h) covers the winning
    column at nearly full image height, and reliable=False means no
    column cleared the minimum bar (caller should refuse to guess).
    """
    h, w = enhanced.shape[:2]
    y0 = int(h * Config.VERTICAL_MARGIN_FRAC)
    y1 = h - y0
    col_w = max(10, int(w * Config.COLUMN_WIDTH_FRAC))
    step = max(4, int(w * Config.COLUMN_SCAN_STEP_FRAC))

    best = None  # (score, x0, x1, n_peaks, periodicity)
    for x0 in range(0, max(1, w - col_w), step):
        x1 = min(w, x0 + col_w)
        profile = _column_profile(enhanced, x0, x1, y0, y1)
        n_peaks, periodicity = _ridge_score(profile)
        score = n_peaks * periodicity
        if best is None or score > best[0]:
            best = (score, x0, x1, n_peaks, periodicity)

    if best is None:
        return (0, y0, w, y1 - y0), False

    score, x0, x1, n_peaks, periodicity = best
    reliable = (n_peaks >= Config.MIN_RIDGE_COUNT) and (periodicity >= Config.MIN_PERIODICITY)

    # Widen a touch from the winning strip toward a sensible stack-width
    # box for a nicer annotation box (visual only, detection strip below
    # still uses the precise center column).
    center = (x0 + x1) // 2
    box_half = int(col_w * 1.6)
    bx0 = max(0, center - box_half)
    bx1 = min(w, center + box_half)
    box = (bx0, y0, bx1 - bx0, y1 - y0)
    return box, reliable, (x0, y0, x1 - x0, y1 - y0), n_peaks, periodicity


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
    if not box_reliable:
        return "Low"
    score = consistency
    if blur < Config.BLUR_WARN_THRESHOLD:
        score -= 0.25
    if brightness < Config.BRIGHTNESS_WARN_LOW or brightness > Config.BRIGHTNESS_WARN_HIGH:
        score -= 0.15
    if n_peaks < 4:
        score -= 0.2
    if score >= 0.6:
        return "High"
    if score >= 0.3:
        return "Medium"
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

    stack_box, box_reliable, strip_box, n_ridge_peaks, periodicity = locate_stack_column(enhanced)

    if not box_reliable:
        # Refuse to guess rather than count background clutter.
        raise ValueError(
            "Couldn't confidently locate a ring stack in this photo. "
            "Try a straighter-on shot with the stack filling more of the "
            "frame, or better separation from background clutter."
        )

    glare = check_glare(gray, stack_box)

    rx, ry, rw, rh = strip_box
    profile = _column_profile(enhanced, rx, rx + rw, ry, ry + rh)
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

    # Stack bounding box (orange)
    cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), (0, 165, 255), 2)
    # Analysis strip (blue) - the actual column used for the signal
    cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (255, 80, 0), 2)
    # Ring lines (green) - only across the stack box width, not full frame
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
    logger.info(
        f"Result: {ring_count} rings | confidence={confidence} | "
        f"blur={blur_score:.1f} | ridge_peaks={n_ridge_peaks} | periodicity={periodicity:.2f}"
    )

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
    except ValueError as e:
        # Expected, user-facing refusal (e.g. couldn't locate stack)
        logger.warning(f"Refused: {e}")
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return jsonify({"error": "Something went wrong processing this photo."}), 500
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Image too large. Please use a photo under 10 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
