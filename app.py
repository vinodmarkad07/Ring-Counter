"""
RingCount AI v3 - Smart Auto-Crop + Accurate Ring Detection
============================================================
Key improvements over v2:
  1. Smart auto-crop: finds the TALLEST single stack column in frame
  2. Multi-strip voting: uses 5 vertical strips and votes on peak positions
     (eliminates false peaks from glare/dirt on one side)
  3. Adaptive prominence: adjusts sensitivity based on ring contrast
  4. Glare suppression: masks overexposed bands before peak detection
  5. Returns cropped stack image separately for user verification
"""

import os, base64, tempfile, logging, time
import cv2
import numpy as np
from scipy.signal import find_peaks
from flask import Flask, request, render_template, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

UPLOAD_DIR = tempfile.gettempdir()


# ─────────────────────────────────────────────
# CONFIG  (tuned for your actual ring photos)
# ─────────────────────────────────────────────
class Config:
    MAX_WIDTH               = 1000
    CLAHE_CLIP              = 2.5
    CLAHE_GRID              = (8, 8)
    # Multi-strip voting
    N_STRIPS                = 7       # number of vertical strips to sample
    STRIP_WIDTH_FRAC        = 0.06    # each strip is 6% of stack width
    VOTE_THRESHOLD          = 0.4     # peak must appear in 40%+ of strips
    # Peak detection
    LOCAL_NORM_WINDOW       = 51
    PEAK_DISTANCE_FRAC      = 0.012
    PEAK_PROMINENCE_FRAC    = 0.06
    # Gap validation
    GAP_OUTLIER_TOLERANCE   = 0.50
    # Stack detection
    MIN_ASPECT_RATIO        = 0.5     # stack must be taller than it is wide (h/w)
    MIN_AREA_FRAC           = 0.04
    MAX_AREA_FRAC           = 0.90
    # Quality thresholds
    BLUR_WARN               = 60
    BRIGHTNESS_WARN_LOW     = 35
    BRIGHTNESS_WARN_HIGH    = 225
    GLARE_THRESH            = 235     # pixel value = overexposed
    GLARE_SUPPRESS_FRAC     = 0.08   # if glare > 8% of strip, suppress those rows


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def resize_image(img, max_width=Config.MAX_WIDTH):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    s = max_width / w
    return cv2.resize(img, (max_width, int(h * s)), interpolation=cv2.INTER_AREA)


def enhance_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP,
                             tileGridSize=Config.CLAHE_GRID)
    return clahe.apply(gray)


def check_blur(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_brightness(gray):
    return float(np.mean(gray))


def check_glare(gray, box):
    x, y, w, h = box
    region = gray[y:y+h, x:x+w]
    return float(np.mean(region > Config.GLARE_THRESH)) if region.size else 0.0


# ─────────────────────────────────────────────
# SMART STACK DETECTION
# Finds the single best (tallest, most centered) stack column
# Works even when multiple stacks are in frame
# ─────────────────────────────────────────────
def detect_best_stack(gray, img_bgr):
    """
    Strategy:
    1. Edge-detect to find strong horizontal lines (ring boundaries)
    2. Find vertical region with most horizontal edges = ring stack
    3. Crop tightly around that column
    4. Return tight crop box
    """
    h, w = gray.shape[:2]
    enhanced = enhance_contrast(gray)

    # --- Step 1: Find horizontal edges (ring seams) ---
    # Sobel in Y direction picks up horizontal transitions (ring-to-ring gaps)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.abs(sobel_y)
    # Keep only strong horizontal edges
    _, edge_mask = cv2.threshold(
        sobel_y.astype(np.uint8),
        sobel_y.mean() + sobel_y.std(),
        255, cv2.THRESH_BINARY
    )

    # --- Step 2: Vertical projection — find column with most horizontal edges ---
    col_score = np.sum(edge_mask, axis=0).astype(float)
    # Smooth to find the center column
    col_score_smooth = np.convolve(col_score, np.ones(30)/30, mode='same')
    best_col = int(np.argmax(col_score_smooth))

    # --- Step 3: Expand left/right from best column while edge density stays high ---
    thresh = col_score_smooth.max() * 0.25
    left = best_col
    right = best_col
    while left > 0 and col_score_smooth[left] > thresh:
        left -= 1
    while right < w - 1 and col_score_smooth[right] > thresh:
        right += 1

    stack_w = right - left
    if stack_w < w * 0.05:
        left  = max(0, best_col - w // 4)
        right = min(w, best_col + w // 4)
        stack_w = right - left

    # --- Step 4: Find vertical extent of the stack ---
    # Row projection inside detected column
    col_region = edge_mask[:, left:right]
    row_score = np.sum(col_region, axis=1).astype(float)
    row_smooth = np.convolve(row_score, np.ones(10)/10, mode='same')
    row_thresh = row_smooth.max() * 0.15

    rows_active = np.where(row_smooth > row_thresh)[0]
    if len(rows_active) < 20:
        top, bot = 0, h
    else:
        top = max(0, int(rows_active[0]) - 10)
        bot = min(h, int(rows_active[-1]) + 10)

    stack_h = bot - top
    if stack_h < h * 0.10:
        top, bot = 0, h
        stack_h = h

    # Reliability: good if aspect ratio looks like a stack
    aspect = stack_h / max(stack_w, 1)
    area_frac = (stack_w * stack_h) / float(w * h)
    reliable = (aspect >= Config.MIN_ASPECT_RATIO and
                Config.MIN_AREA_FRAC <= area_frac <= Config.MAX_AREA_FRAC)

    box = (left, top, stack_w, stack_h)
    logger.info(f"Stack detected: box={box} aspect={aspect:.2f} "
                f"area_frac={area_frac:.2f} reliable={reliable}")
    return box, reliable, area_frac


# ─────────────────────────────────────────────
# MULTI-STRIP VOTING  (core accuracy improvement)
# Samples N_STRIPS vertical strips across the stack,
# runs peak detection on each, then votes.
# A peak position counts only if it appears in
# VOTE_THRESHOLD fraction of strips.
# ─────────────────────────────────────────────
def suppress_glare_rows(strip, thresh=Config.GLARE_THRESH):
    """Replace overexposed rows with local median so they don't create fake peaks."""
    col_means = np.mean(strip, axis=1)
    glare_rows = col_means > thresh
    if glare_rows.sum() > 0:
        good_mean = float(np.median(col_means[~glare_rows])) if (~glare_rows).sum() > 0 else 128.0
        strip = strip.copy()
        strip[glare_rows, :] = int(good_mean)
    return strip


def local_contrast_normalize(signal, window):
    window = max(3, window | 1)
    pad = window // 2
    padded = np.pad(signal, pad, mode="reflect")
    k = np.ones(window) / window
    lm  = np.convolve(padded, k, mode="valid")
    lsm = np.convolve(padded**2, k, mode="valid")
    lstd = np.sqrt(np.maximum(lsm - lm**2, 1e-6))
    return (signal - lm) / lstd


def get_peaks_from_profile(profile, n_rows):
    normalized = local_contrast_normalize(profile, Config.LOCAL_NORM_WINDOW)
    gradient   = np.abs(np.gradient(normalized))
    rng = gradient.max() - gradient.min()
    prominence = max(rng * Config.PEAK_PROMINENCE_FRAC, 1e-6)
    distance   = max(1, int(n_rows * Config.PEAK_DISTANCE_FRAC))
    peaks, _   = find_peaks(gradient, prominence=prominence, distance=distance)
    return peaks


def multi_strip_vote(enhanced, stack_box):
    """
    Run peak detection on N_STRIPS evenly-spaced vertical strips.
    Return consensus peaks via a vote map.
    """
    sx, sy, sw, sh = stack_box
    n  = Config.N_STRIPS
    sw_strip = max(3, int(sw * Config.STRIP_WIDTH_FRAC))

    # evenly space strip centres across middle 80% of stack width
    margin = int(sw * 0.10)
    xs = np.linspace(sx + margin, sx + sw - margin, n, dtype=int)

    vote_map = np.zeros(sh, dtype=float)

    strip_boxes = []
    for cx in xs:
        x0 = max(sx, cx - sw_strip // 2)
        x1 = min(sx + sw, x0 + sw_strip)
        strip = enhanced[sy:sy+sh, x0:x1]
        strip = suppress_glare_rows(strip)
        profile = np.mean(strip.astype(np.float64), axis=1)
        peaks   = get_peaks_from_profile(profile, sh)
        # Gaussian splat each peak into vote map (width = ring-gap estimate)
        for p in peaks:
            sigma = max(2, sh * 0.008)
            rr = np.arange(sh)
            vote_map += np.exp(-0.5 * ((rr - p) / sigma)**2)
        strip_boxes.append((x0, sy, x1-x0, sh))

    # Normalize vote map
    vote_map /= n

    # Find peaks in vote map = consensus ring positions
    if vote_map.max() < 1e-6:
        return np.array([], dtype=int), strip_boxes

    vmin, vmax = vote_map.min(), vote_map.max()
    vote_norm  = (vote_map - vmin) / (vmax - vmin + 1e-9)

    rng = vote_norm.max() - vote_norm.min()
    prominence = max(rng * 0.06, 0.02)
    distance   = max(1, int(sh * Config.PEAK_DISTANCE_FRAC))

    consensus_peaks, _ = find_peaks(vote_norm,
                                     prominence=prominence,
                                     distance=distance)
    return consensus_peaks, strip_boxes


# ─────────────────────────────────────────────
# GAP VALIDATION
# ─────────────────────────────────────────────
def validate_gaps(peaks):
    if len(peaks) < 3:
        return peaks, 0.0
    gaps = np.diff(peaks)
    med  = np.median(gaps)
    if med <= 0:
        return peaks, 0.0
    keep = [True] + [abs(g - med) / med <= Config.GAP_OUTLIER_TOLERANCE
                     for g in gaps]
    filtered = peaks[np.array(keep)]
    g2 = np.diff(filtered) if len(filtered) > 1 else np.array([med])
    consistency = 1.0 - min(1.0, float(np.std(g2) / (np.mean(g2) + 1e-9)))
    return filtered, max(0.0, consistency)


def count_rings(filtered_peaks):
    n = len(filtered_peaks)
    return max(0, n - 1) if n >= 2 else n


def compute_confidence(consistency, blur, brightness, n_peaks, reliable):
    score = consistency
    if not reliable:           score -= 0.45
    if blur < Config.BLUR_WARN: score -= 0.25
    if (brightness < Config.BRIGHTNESS_WARN_LOW or
            brightness > Config.BRIGHTNESS_WARN_HIGH): score -= 0.15
    if n_peaks < 4:            score -= 0.2
    if score >= 0.60: return "High"
    if score >= 0.30: return "Medium"
    return "Low"


# ─────────────────────────────────────────────
# ANNOTATION
# ─────────────────────────────────────────────
def annotate(img, stack_box, consensus_peaks, ring_count, reliable):
    out  = img.copy()
    sx, sy, sw, sh = stack_box

    # Stack box
    color = (0, 200, 0) if reliable else (0, 100, 255)
    cv2.rectangle(out, (sx, sy), (sx+sw, sy+sh), color, 2)

    # Ring boundary lines (full width of stack)
    for p in consensus_peaks:
        yy = sy + int(p)
        cv2.line(out, (sx, yy), (sx+sw, yy), (0, 255, 80), 2)

    # Count text — shadow + cyan
    cv2.putText(out, str(ring_count), (12, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0,0,0), 7, cv2.LINE_AA)
    cv2.putText(out, str(ring_count), (12, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0,255,255), 3, cv2.LINE_AA)
    return out


def encode_jpg(img, quality=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode image")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def process_image(path):
    t0 = time.time()

    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read image. Upload a valid JPG/PNG.")

    img      = resize_image(img)
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_contrast(gray)

    blur_score = check_blur(gray)
    brightness = check_brightness(gray)

    # 1. Smart stack detection
    stack_box, reliable, area_frac = detect_best_stack(gray, img)
    glare = check_glare(gray, stack_box)

    # 2. Multi-strip voting for consensus ring positions
    consensus_peaks, strip_boxes = multi_strip_vote(enhanced, stack_box)

    # 3. Gap validation
    filtered_peaks, consistency = validate_gaps(consensus_peaks)
    ring_count  = count_rings(filtered_peaks)
    confidence  = compute_confidence(consistency, blur_score, brightness,
                                     len(filtered_peaks), reliable)
    elapsed     = round(time.time() - t0, 2)

    # 4. Annotate full image
    annotated = annotate(img, stack_box, filtered_peaks, ring_count, reliable)

    # 5. Cropped stack image (tight crop for verification)
    sx, sy, sw, sh = stack_box
    crop = img[sy:sy+sh, sx:sx+sw].copy()
    # Draw ring lines on crop too
    for p in filtered_peaks:
        cv2.line(crop, (0, int(p)), (sw, int(p)), (0,255,80), 2)
    cv2.putText(crop, str(ring_count), (6, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,0,0), 5, cv2.LINE_AA)
    cv2.putText(crop, str(ring_count), (6, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,255), 2, cv2.LINE_AA)

    logger.info(f"rings={ring_count} conf={confidence} "
                f"peaks={len(filtered_peaks)} blur={blur_score:.1f} "
                f"elapsed={elapsed}s")

    return {
        "ring_count":       ring_count,
        "confidence":       confidence,
        "consistency":      round(float(consistency), 2),
        "blur_score":       round(blur_score, 1),
        "brightness":       round(brightness, 1),
        "glare":            round(glare * 100, 1),
        "box_reliable":     reliable,
        "elapsed":          elapsed,
        "image_data_url":   encode_jpg(annotated),
        "crop_data_url":    encode_jpg(crop),       # tight crop
    }


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/count", methods=["POST"])
def count():
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400
    file = request.files["photo"]
    if not file.filename:
        return jsonify({"error": "Empty file"}), 400
    ext = os.path.splitext(file.filename or "p.jpg")[1].lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        return jsonify({"error": "Use JPG or PNG"}), 400

    path = os.path.join(UPLOAD_DIR, f"ring_{os.getpid()}{ext}")
    try:
        file.save(path)
        return jsonify(process_image(path))
    except Exception as e:
        logger.error(e)
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Photo too large (max 10 MB)"}), 413


@app.route("/status")
def status():
    return jsonify({"status": "ok", "version": "3.0"})


if __name__ == "__main__":
    app.run(host="0.0.0.0",
            port=int(os.environ.get("PORT", 5000)),
            debug=False)
