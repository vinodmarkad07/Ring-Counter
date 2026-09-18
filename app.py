"""
RingCount AI v4 — Manual-Axis Edition
============================================================
Why this version exists:
  v3 auto-detected the stack boundary using Sobel edge density.
  On real client photos it locked onto ceiling rafters, glare
  streaks, and cardboard tags instead of the ring stack, because
  ANY strong horizontal edge scored the same as a ring seam.
  That is a structural failure, not a tuning problem — no
  amount of peak-detection tweaking fixes a wrong input region.

  v4 removes auto stack-detection entirely. The user taps the
  TOP and BOTTOM of the stack on the photo (2 taps, ~2 seconds).
  This guarantees the measuring axis is always inside the real
  stack, which is what actually drives accuracy. Peak detection
  and gap validation are also tightened so a bad/ambiguous photo
  reports LOW confidence instead of silently guessing.

  This does not promise 100% accuracy on every photo — no vision
  system can promise that on damaged/warped/dirty stacks. It
  promises that when the algorithm reports HIGH confidence, the
  count is trustworthy, and when it isn't, it tells you honestly
  instead of hiding it.
"""

import os, base64, tempfile, logging, time
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify


# Pure numpy replacement for scipy.signal.find_peaks
# (removes scipy dependency so Vercel deploys reliably)
def find_peaks(x, prominence=0.0, distance=1):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return np.array([], dtype=int), {}
    # Local maxima
    peaks = [i for i in range(1, n-1) if x[i] > x[i-1] and x[i] > x[i+1]]
    peaks = np.array(peaks, dtype=int)
    if len(peaks) == 0:
        return peaks, {}
    # Distance filter (keep highest in each window)
    if distance > 1 and len(peaks) > 1:
        keep = np.ones(len(peaks), dtype=bool)
        for i in range(len(peaks)):
            if not keep[i]:
                continue
            for j in range(i+1, len(peaks)):
                if peaks[j] - peaks[i] < distance:
                    if x[peaks[j]] >= x[peaks[i]]:
                        keep[i] = False; break
                    else:
                        keep[j] = False
                else:
                    break
        peaks = peaks[keep]
    # Prominence filter
    if prominence > 0 and len(peaks) > 0:
        kept = []
        for p in peaks:
            lm = x[p]
            for i in range(p-1, -1, -1):
                if x[i] < lm: lm = x[i]
                if x[i] > x[p]: break
            rm = x[p]
            for i in range(p+1, n):
                if x[i] < rm: rm = x[i]
                if x[i] > x[p]: break
            if x[p] - max(lm, rm) >= prominence:
                kept.append(p)
        peaks = np.array(kept, dtype=int)
    return peaks, {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

UPLOAD_DIR = tempfile.gettempdir()


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
class Config:
    MAX_WIDTH               = 1200
    CLAHE_CLIP               = 2.5
    CLAHE_GRID                = (8, 8)
    N_STRIPS                  = 9
    STRIP_WIDTH_FRAC          = 0.06
    AXIS_HALF_WIDTH_FRAC      = 0.22
    LOCAL_NORM_WINDOW         = 41
    PEAK_DISTANCE_FRAC        = 0.015
    PEAK_PROMINENCE_FRAC      = 0.09
    VOTE_PROMINENCE_FRAC      = 0.10
    GAP_OUTLIER_TOLERANCE     = 0.35
    MIN_STRIPS_AGREEING       = 0.55
    BLUR_WARN                 = 60
    BRIGHTNESS_WARN_LOW       = 35
    BRIGHTNESS_WARN_HIGH      = 225
    GLARE_THRESH               = 235
    GLARE_SUPPRESS_FRAC        = 0.08


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def resize_image(img, max_width=Config.MAX_WIDTH):
    h, w = img.shape[:2]
    if w <= max_width:
        return img, 1.0
    s = max_width / w
    return cv2.resize(img, (max_width, int(h * s)), interpolation=cv2.INTER_AREA), s


def enhance_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP, tileGridSize=Config.CLAHE_GRID)
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
# MULTI-STRIP VOTING (runs strictly inside the user-tapped axis box)
# ─────────────────────────────────────────────
def suppress_glare_rows(strip, thresh=Config.GLARE_THRESH):
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


def multi_strip_vote(enhanced, axis_box):
    """axis_box is the user-defined column — guaranteed inside the real stack."""
    sx, sy, sw, sh = axis_box
    n  = Config.N_STRIPS
    sw_strip = max(3, int(sw * Config.STRIP_WIDTH_FRAC))

    margin = int(sw * 0.08)
    xs = np.linspace(sx + margin, sx + sw - margin, n, dtype=int)

    vote_map = np.zeros(sh, dtype=float)
    strips_hit = np.zeros(sh, dtype=int)

    for cx in xs:
        x0 = max(sx, cx - sw_strip // 2)
        x1 = min(sx + sw, x0 + sw_strip)
        strip = enhanced[sy:sy+sh, x0:x1]
        strip = suppress_glare_rows(strip)
        profile = np.mean(strip.astype(np.float64), axis=1)
        peaks   = get_peaks_from_profile(profile, sh)
        sigma = max(2, sh * 0.007)
        rr = np.arange(sh)
        for p in peaks:
            vote_map += np.exp(-0.5 * ((rr - p) / sigma)**2)
            strips_hit[max(0, p-2):min(sh, p+3)] += 1

    vote_map /= n

    if vote_map.max() < 1e-6:
        return np.array([], dtype=int), 0

    vmin, vmax = vote_map.min(), vote_map.max()
    vote_norm  = (vote_map - vmin) / (vmax - vmin + 1e-9)

    rng = vote_norm.max() - vote_norm.min()
    prominence = max(rng * Config.VOTE_PROMINENCE_FRAC, 0.03)
    distance   = max(1, int(sh * Config.PEAK_DISTANCE_FRAC))

    candidate_peaks, _ = find_peaks(vote_norm, prominence=prominence, distance=distance)

    # Reject peaks that fewer than MIN_STRIPS_AGREEING of strips actually voted for.
    # A peak caused by clutter on one side of the frame (a rafter, a reflection) only
    # shows up in 1-2 strips and gets dropped here — this is the key fix vs v3.
    min_hits = max(1, int(np.ceil(n * Config.MIN_STRIPS_AGREEING)))
    confirmed = np.array([p for p in candidate_peaks if strips_hit[p] >= min_hits], dtype=int)

    rejected_count = len(candidate_peaks) - len(confirmed)
    return confirmed, rejected_count


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
    keep = [True] + [abs(g - med) / med <= Config.GAP_OUTLIER_TOLERANCE for g in gaps]
    filtered = peaks[np.array(keep)]
    g2 = np.diff(filtered) if len(filtered) > 1 else np.array([med])
    consistency = 1.0 - min(1.0, float(np.std(g2) / (np.mean(g2) + 1e-9)))
    return filtered, max(0.0, consistency)


def count_rings(filtered_peaks):
    n = len(filtered_peaks)
    return max(0, n - 1) if n >= 2 else n


def compute_confidence(consistency, blur, brightness, n_peaks, rejected_count):
    score = consistency
    if blur < Config.BLUR_WARN:                score -= 0.25
    if (brightness < Config.BRIGHTNESS_WARN_LOW or
            brightness > Config.BRIGHTNESS_WARN_HIGH): score -= 0.15
    if n_peaks < 4:                              score -= 0.25
    if rejected_count > n_peaks * 0.3:            score -= 0.15
    if score >= 0.65: return "High"
    if score >= 0.35: return "Medium"
    return "Low"


# ─────────────────────────────────────────────
# ANNOTATION
# ─────────────────────────────────────────────
def annotate(img, axis_box, confirmed_peaks, ring_count):
    out  = img.copy()
    sx, sy, sw, sh = axis_box

    cv2.rectangle(out, (sx, sy), (sx+sw, sy+sh), (0, 200, 0), 2)

    for p in confirmed_peaks:
        yy = sy + int(p)
        cv2.line(out, (sx, yy), (sx+sw, yy), (0, 255, 80), 2)

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
# MAIN PIPELINE — requires top_frac / bottom_frac / x_frac from the user's taps
# ─────────────────────────────────────────────
def process_image(path, top_frac, bottom_frac, x_frac):
    t0 = time.time()

    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read image. Upload a valid JPG/PNG.")

    img, scale = resize_image(img)
    h, w = img.shape[:2]
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_contrast(gray)

    blur_score = check_blur(gray)
    brightness = check_brightness(gray)

    top_y    = int(top_frac * h)
    bottom_y = int(bottom_frac * h)
    if bottom_y <= top_y + 20:
        raise ValueError("Bottom tap must be clearly below the top tap.")
    center_x = int(x_frac * w)
    axis_h   = bottom_y - top_y
    axis_w   = max(20, int(axis_h * Config.AXIS_HALF_WIDTH_FRAC * 2))
    axis_x   = max(0, center_x - axis_w // 2)
    axis_w   = min(axis_w, w - axis_x)
    axis_box = (axis_x, top_y, axis_w, axis_h)

    glare = check_glare(gray, axis_box)

    confirmed_peaks, rejected_count = multi_strip_vote(enhanced, axis_box)
    filtered_peaks, consistency = validate_gaps(confirmed_peaks)
    ring_count  = count_rings(filtered_peaks)
    confidence  = compute_confidence(consistency, blur_score, brightness,
                                      len(filtered_peaks), rejected_count)
    elapsed     = round(time.time() - t0, 2)

    annotated = annotate(img, axis_box, filtered_peaks, ring_count)

    sx, sy, sw, sh = axis_box
    pad_x = int(sw * 1.5)
    cx0, cx1 = max(0, sx - pad_x), min(w, sx + sw + pad_x)
    crop = img[sy:sy+sh, cx0:cx1].copy()
    rel_x = sx - cx0
    for p in filtered_peaks:
        cv2.line(crop, (0, int(p)), (crop.shape[1], int(p)), (0,255,80), 2)
    cv2.rectangle(crop, (rel_x, 0), (rel_x+sw, sh), (0, 200, 0), 2)
    cv2.putText(crop, str(ring_count), (6, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,0,0), 5, cv2.LINE_AA)
    cv2.putText(crop, str(ring_count), (6, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,255), 2, cv2.LINE_AA)

    logger.info(f"rings={ring_count} conf={confidence} peaks={len(filtered_peaks)} "
                f"rejected={rejected_count} blur={blur_score:.1f} elapsed={elapsed}s")

    return {
        "ring_count":       ring_count,
        "confidence":       confidence,
        "consistency":      round(float(consistency), 2),
        "blur_score":       round(blur_score, 1),
        "brightness":       round(brightness, 1),
        "glare":            round(glare * 100, 1),
        "rejected_peaks":   int(rejected_count),
        "elapsed":          elapsed,
        "image_data_url":   encode_jpg(annotated),
        "crop_data_url":    encode_jpg(crop),
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

    try:
        top_frac    = float(request.form.get("top_frac", ""))
        bottom_frac = float(request.form.get("bottom_frac", ""))
        x_frac      = float(request.form.get("x_frac", "0.5"))
    except (TypeError, ValueError):
        return jsonify({"error": "Missing tap coordinates. Tap the top and bottom of the stack first."}), 400

    if not (0 <= top_frac < bottom_frac <= 1):
        return jsonify({"error": "Invalid tap positions. Tap top of stack, then bottom of stack."}), 400

    path = os.path.join(UPLOAD_DIR, f"ring_{os.getpid()}_{int(time.time()*1000)}{ext}")
    try:
        file.save(path)
        return jsonify(process_image(path, top_frac, bottom_frac, x_frac))
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
    return jsonify({"status": "ok", "version": "4.0-manual-axis"})


if __name__ == "__main__":
    app.run(host="0.0.0.0",
            port=int(os.environ.get("PORT", 5000)),
            debug=False)
