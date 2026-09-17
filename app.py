"""
RingCount AI v5
Robust ring counting for cylindrical metal ring stacks.

Pipeline:
1. User marks TOP and BOTTOM of stack.
2. Build a vertical measuring region.
3. Normalize illumination.
4. Measure horizontal-edge strength.
5. Run multiple vertical strips.
6. Vote across strips.
7. Detect candidate ring seams.
8. Remove isolated/reflection peaks.
9. Validate spacing.
10. Return count + annotated image.

No SciPy dependency.
Designed for Flask/Vercel deployment.
"""

import os
import base64
import tempfile
import logging
import time

import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify


# ============================================================
# APP
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Browser compression normally keeps this below the limit.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

UPLOAD_DIR = tempfile.gettempdir()


# ============================================================
# CONFIGURATION
# ============================================================

class Config:

    # Image
    MAX_WIDTH = 1400

    # Axis width
    AXIS_WIDTH_FRAC = 0.34

    # Number of vertical strips
    N_STRIPS = 13

    # Strip width relative to measuring region
    STRIP_WIDTH_FRAC = 0.075

    # Ignore pixels close to left/right axis boundaries
    SIDE_MARGIN_FRAC = 0.10

    # Signal processing
    SMOOTH_KERNEL = 5
    LOCAL_WINDOW = 41

    # Peak detection
    MIN_DISTANCE_FRAC = 0.007
    MAX_DISTANCE_FRAC = 0.035

    # Voting
    MIN_STRIP_AGREEMENT = 0.45

    # Spacing validation
    MAX_GAP_DEVIATION = 0.45

    # Image quality
    BLUR_WARN = 45
    GLARE_THRESHOLD = 245

    # Very small peaks are ignored
    MIN_SIGNAL_LEVEL = 0.08


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def resize_image(img, max_width=Config.MAX_WIDTH):

    h, w = img.shape[:2]

    if w <= max_width:
        return img

    scale = max_width / float(w)

    return cv2.resize(
        img,
        (
            max_width,
            int(h * scale)
        ),
        interpolation=cv2.INTER_AREA
    )


def encode_jpg(img, quality=88):

    ok, buf = cv2.imencode(
        ".jpg",
        img,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )

    if not ok:
        raise ValueError("Could not encode result image")

    return (
        "data:image/jpeg;base64,"
        + base64.b64encode(buf).decode("ascii")
    )


# ============================================================
# IMAGE QUALITY
# ============================================================

def blur_score(gray):

    return float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )


def brightness_score(gray):

    return float(np.mean(gray))


def glare_score(gray, box):

    x, y, w, h = box

    region = gray[
        max(0, y):min(gray.shape[0], y + h),
        max(0, x):min(gray.shape[1], x + w)
    ]

    if region.size == 0:
        return 0.0

    return float(
        np.mean(region >= Config.GLARE_THRESHOLD)
    )


# ============================================================
# STACK AXIS
# ============================================================

def build_axis_box(gray, top_frac, bottom_frac, x_frac):

    h, w = gray.shape[:2]

    top_y = int(top_frac * h)
    bottom_y = int(bottom_frac * h)

    if bottom_y <= top_y + 30:
        raise ValueError(
            "Top and bottom taps are too close together."
        )

    center_x = int(x_frac * w)

    axis_h = bottom_y - top_y

    axis_w = max(
        30,
        int(axis_h * Config.AXIS_WIDTH_FRAC)
    )

    axis_x = center_x - axis_w // 2

    axis_x = max(
        0,
        min(axis_x, w - 1)
    )

    axis_w = min(
        axis_w,
        w - axis_x
    )

    return (
        axis_x,
        top_y,
        axis_w,
        axis_h
    )


# ============================================================
# PREPROCESSING
# ============================================================

def prepare_gray(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # Small blur removes camera noise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    return gray


# ============================================================
# LOCAL NORMALIZATION
# ============================================================

def normalize_signal(signal):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    n = len(signal)

    if n < 5:
        return signal

    window = min(
        Config.LOCAL_WINDOW,
        n if n % 2 else n - 1
    )

    window = max(
        5,
        window
    )

    if window >= n:
        window = n - 1 if n % 2 == 0 else n

    kernel = np.ones(
        window,
        dtype=np.float64
    ) / window

    padded = np.pad(
        signal,
        window // 2,
        mode="reflect"
    )

    mean = np.convolve(
        padded,
        kernel,
        mode="valid"
    )

    sq_mean = np.convolve(
        padded ** 2,
        kernel,
        mode="valid"
    )

    variance = np.maximum(
        sq_mean - mean ** 2,
        1e-6
    )

    std = np.sqrt(variance)

    return (signal - mean) / std


# ============================================================
# NUMPY PEAK DETECTOR
# ============================================================

def detect_peaks(signal, min_distance, min_prominence):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    if len(signal) < 3:
        return np.array([], dtype=int)

    # --------------------------------------------------------
    # Candidate local maxima
    # --------------------------------------------------------

    candidates = np.where(
        (signal[1:-1] >= signal[:-2]) &
        (signal[1:-1] > signal[2:])
    )[0] + 1

    if len(candidates) == 0:
        return np.array([], dtype=int)

    # --------------------------------------------------------
    # Approximate prominence
    # --------------------------------------------------------

    prominences = []

    search_radius = max(
        min_distance * 3,
        12
    )

    for p in candidates:

        left_start = max(
            0,
            p - search_radius
        )

        right_end = min(
            len(signal),
            p + search_radius + 1
        )

        left = signal[
            left_start:p
        ]

        right = signal[
            p + 1:right_end
        ]

        if len(left):
            left_min = np.min(left)
        else:
            left_min = signal[p]

        if len(right):
            right_min = np.min(right)
        else:
            right_min = signal[p]

        base = max(
            left_min,
            right_min
        )

        prominence = signal[p] - base

        prominences.append(
            prominence
        )

    prominences = np.asarray(
        prominences
    )

    keep = prominences >= min_prominence

    candidates = candidates[keep]
    prominences = prominences[keep]

    if len(candidates) == 0:
        return np.array([], dtype=int)

    # --------------------------------------------------------
    # Non-maximum suppression
    # --------------------------------------------------------

    order = np.argsort(
        prominences
    )[::-1]

    selected = []

    for idx in order:

        p = int(candidates[idx])

        too_close = False

        for q in selected:

            if abs(p - q) < min_distance:
                too_close = True
                break

        if not too_close:
            selected.append(p)

    selected.sort()

    return np.asarray(
        selected,
        dtype=int
    )


# ============================================================
# STRIP SIGNAL
# ============================================================

def strip_signal(gray, x0, x1, y0, y1):

    roi = gray[
        y0:y1,
        x0:x1
    ]

    if roi.size == 0:
        return np.zeros(
            max(1, y1 - y0),
            dtype=np.float64
        )

    # --------------------------------------------------------
    # Vertical Sobel = horizontal edge detector
    #
    # Ring seams are predominantly horizontal.
    # Therefore Sobel-Y is more appropriate than generic
    # intensity changes.
    # --------------------------------------------------------

    sobel_y = cv2.Sobel(
        roi,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    edge_strength = np.abs(
        sobel_y
    )

    # Median is robust against isolated reflections.
    signal = np.median(
        edge_strength,
        axis=1
    )

    # Normalize local illumination
    signal = normalize_signal(
        signal
    )

    # Smooth
    kernel = np.ones(
        Config.SMOOTH_KERNEL,
        dtype=np.float64
    ) / Config.SMOOTH_KERNEL

    signal = np.convolve(
        np.pad(
            signal,
            Config.SMOOTH_KERNEL // 2,
            mode="reflect"
        ),
        kernel,
        mode="valid"
    )

    return signal


# ============================================================
# MULTI STRIP DETECTION
# ============================================================

def multi_strip_detection(gray, axis_box):

    sx, sy, sw, sh = axis_box

    n = Config.N_STRIPS

    margin = int(
        sw * Config.SIDE_MARGIN_FRAC
    )

    left = sx + margin
    right = sx + sw - margin

    if right <= left:
        return np.array([], dtype=int), 0

    centers = np.linspace(
        left,
        right,
        n
    ).astype(int)

    strip_width = max(
        4,
        int(sw * Config.STRIP_WIDTH_FRAC)
    )

    all_peaks = []

    min_distance = max(
        3,
        int(sh * Config.MIN_DISTANCE_FRAC)
    )

    max_distance = max(
        min_distance + 1,
        int(sh * Config.MAX_DISTANCE_FRAC)
    )

    for cx in centers:

        x0 = max(
            sx,
            cx - strip_width // 2
        )

        x1 = min(
            sx + sw,
            cx + strip_width // 2
        )

        signal = strip_signal(
            gray,
            x0,
            x1,
            sy,
            sy + sh
        )

        if len(signal) < 10:
            continue

        # Robust adaptive threshold
        median = np.median(signal)
        mad = np.median(
            np.abs(signal - median)
        ) + 1e-6

        prominence = max(
            Config.MIN_SIGNAL_LEVEL,
            mad * 1.8
        )

        peaks = detect_peaks(
            signal,
            min_distance,
            prominence
        )

        # Keep plausible peaks
        peaks = peaks[
            (peaks > 3) &
            (peaks < sh - 4)
        ]

        for p in peaks:
            all_peaks.append(
                int(p)
            )

    if not all_peaks:
        return np.array([], dtype=int), 0

    # --------------------------------------------------------
    # Cluster peaks from different strips
    # --------------------------------------------------------

    all_peaks.sort()

    cluster_radius = max(
        3,
        int(sh * 0.006)
    )

    clusters = []

    current = [
        all_peaks[0]
    ]

    for p in all_peaks[1:]:

        if abs(
            p - np.median(current)
        ) <= cluster_radius:

            current.append(p)

        else:

            clusters.append(
                current
            )

            current = [p]

    clusters.append(current)

    # --------------------------------------------------------
    # Require cross-strip agreement
    # --------------------------------------------------------

    min_hits = max(
        2,
        int(
            np.ceil(
                n * Config.MIN_STRIP_AGREEMENT
            )
        )
    )

    confirmed = []

    for cluster in clusters:

        # A cluster can contain multiple detections
        # from neighboring strips.
        hits = len(cluster)

        if hits >= min_hits:

            confirmed.append(
                int(round(np.median(cluster)))
            )

    # Remove duplicates
    if confirmed:

        clean = [
            confirmed[0]
        ]

        for p in confirmed[1:]:

            if p - clean[-1] >= cluster_radius:

                clean.append(p)

        confirmed = clean

    rejected = max(
        0,
        len(clusters) - len(confirmed)
    )

    return (
        np.asarray(
            confirmed,
            dtype=int
        ),
        rejected
    )


# ============================================================
# SPACING VALIDATION
# ============================================================

def validate_ring_spacing(peaks):

    peaks = np.asarray(
        peaks,
        dtype=int
    )

    if len(peaks) < 4:
        return peaks, 0.0

    gaps = np.diff(
        peaks
    ).astype(float)

    median_gap = float(
        np.median(gaps)
    )

    if median_gap <= 0:
        return np.array([], dtype=int), 0.0

    # --------------------------------------------------------
    # Reject impossible gaps
    # --------------------------------------------------------

    valid = [
        True
    ]

    for gap in gaps:

        deviation = abs(
            gap - median_gap
        ) / median_gap

        valid.append(
            deviation <= Config.MAX_GAP_DEVIATION
        )

    filtered = peaks[
        np.asarray(valid)
    ]

    if len(filtered) < 3:
        return peaks, 0.0

    final_gaps = np.diff(
        filtered
    ).astype(float)

    mean_gap = np.mean(
        final_gaps
    )

    std_gap = np.std(
        final_gaps
    )

    consistency = 1.0 - min(
        1.0,
        std_gap / (
            mean_gap + 1e-6
        )
    )

    return (
        filtered,
        max(
            0.0,
            float(consistency)
        )
    )


# ============================================================
# COUNT
# ============================================================

def count_rings(peaks):

    if len(peaks) < 2:
        return 0

    # Boundary count - 1
    return max(
        0,
        int(len(peaks) - 1)
    )


# ============================================================
# CONFIDENCE
# ============================================================

def confidence_level(
    consistency,
    blur,
    brightness,
    glare,
    peak_count,
    rejected
):

    score = 0.0

    # Spacing
    score += consistency * 0.45

    # Peak count
    if peak_count >= 10:
        score += 0.20
    elif peak_count >= 5:
        score += 0.12

    # Sharpness
    if blur >= Config.BLUR_WARN:
        score += 0.15

    # Brightness
    if 40 <= brightness <= 220:
        score += 0.10

    # Glare
    if glare < 0.08:
        score += 0.10
    elif glare > 0.20:
        score -= 0.10

    # Rejected peaks
    if peak_count > 0:
        rejection_ratio = rejected / max(
            1,
            peak_count + rejected
        )

        if rejection_ratio > 0.40:
            score -= 0.15

    if score >= 0.70:
        return "High"

    if score >= 0.45:
        return "Medium"

    return "Low"


# ============================================================
# ANNOTATION
# ============================================================

def annotate(
    img,
    axis_box,
    peaks,
    ring_count,
    confidence
):

    out = img.copy()

    sx, sy, sw, sh = axis_box

    # Stack region
    cv2.rectangle(
        out,
        (sx, sy),
        (sx + sw, sy + sh),
        (0, 220, 0),
        2
    )

    # Ring lines
    for p in peaks:

        y = sy + int(p)

        cv2.line(
            out,
            (sx, y),
            (sx + sw, y),
            (0, 255, 80),
            2
        )

    # Count
    text = f"{ring_count}"

    cv2.putText(
        out,
        text,
        (15, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.3,
        (0, 0, 0),
        7,
        cv2.LINE_AA
    )

    cv2.putText(
        out,
        text,
        (15, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.3,
        (0, 255, 255),
        3,
        cv2.LINE_AA
    )

    # Confidence
    cv2.putText(
        out,
        confidence,
        (15, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 80),
        2,
        cv2.LINE_AA
    )

    return out


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    path,
    top_frac,
    bottom_frac,
    x_frac
):

    start = time.time()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    img = cv2.imread(
        path
    )

    if img is None:
        raise ValueError(
            "Could not read image."
        )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    img = resize_image(
        img
    )

    h, w = img.shape[:2]

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = prepare_gray(
        img
    )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    blur = blur_score(
        gray
    )

    brightness = brightness_score(
        gray
    )

    # --------------------------------------------------------
    # User-selected stack
    # --------------------------------------------------------

    axis_box = build_axis_box(
        gray,
        top_frac,
        bottom_frac,
        x_frac
    )

    glare = glare_score(
        gray,
        axis_box
    )

    # --------------------------------------------------------
    # Detect seams
    # --------------------------------------------------------

    peaks, rejected = multi_strip_detection(
        gray,
        axis_box
    )

    # --------------------------------------------------------
    # Validate spacing
    # --------------------------------------------------------

    filtered_peaks, consistency = validate_ring_spacing(
        peaks
    )

    # --------------------------------------------------------
    # Count
    # --------------------------------------------------------

    ring_count = count_rings(
        filtered_peaks
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = confidence_level(
        consistency,
        blur,
        brightness,
        glare,
        len(filtered_peaks),
        rejected
    )

    elapsed = round(
        time.time() - start,
        2
    )

    # --------------------------------------------------------
    # Annotated image
    # --------------------------------------------------------

    annotated = annotate(
        img,
        axis_box,
        filtered_peaks,
        ring_count,
        confidence
    )

    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    sx, sy, sw, sh = axis_box

    padding = int(
        sw * 1.5
    )

    cx0 = max(
        0,
        sx - padding
    )

    cx1 = min(
        w,
        sx + sw + padding
    )

    crop = img[
        sy:sy + sh,
        cx0:cx1
    ].copy()

    relative_x = sx - cx0

    # Ring lines on crop
    for p in filtered_peaks:

        yy = int(p)

        cv2.line(
            crop,
            (0, yy),
            (crop.shape[1], yy),
            (0, 255, 80),
            2
        )

    cv2.rectangle(
        crop,
        (relative_x, 0),
        (relative_x + sw, sh),
        (0, 220, 0),
        2
    )

    cv2.putText(
        crop,
        str(ring_count),
        (8, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 0, 0),
        5,
        cv2.LINE_AA
    )

    cv2.putText(
        crop,
        str(ring_count),
        (8, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    logger.info(
        "rings=%s confidence=%s peaks=%s "
        "rejected=%s consistency=%.2f "
        "blur=%.1f glare=%.1f%% time=%.2fs",
        ring_count,
        confidence,
        len(filtered_peaks),
        rejected,
        consistency,
        blur,
        glare * 100,
        elapsed
    )

    return {
        "ring_count": ring_count,
        "confidence": confidence,
        "consistency": round(
            float(consistency),
            2
        ),
        "blur_score": round(
            blur,
            1
        ),
        "brightness": round(
            brightness,
            1
        ),
        "glare": round(
            glare * 100,
            1
        ),
        "rejected_peaks": int(
            rejected
        ),
        "elapsed": elapsed,
        "image_data_url": encode_jpg(
            annotated
        ),
        "crop_data_url": encode_jpg(
            crop
        )
    }


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route(
    "/count",
    methods=["POST"]
)
def count():

    if "photo" not in request.files:

        return jsonify({
            "error": "No photo uploaded"
        }), 400

    file = request.files[
        "photo"
    ]

    if not file.filename:

        return jsonify({
            "error": "Empty file"
        }), 400

    ext = os.path.splitext(
        file.filename
    )[1].lower()

    if ext not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }:

        return jsonify({
            "error": "Use JPG, JPEG, PNG or WEBP"
        }), 400

    try:

        top_frac = float(
            request.form.get(
                "top_frac",
                ""
            )
        )

        bottom_frac = float(
            request.form.get(
                "bottom_frac",
                ""
            )
        )

        x_frac = float(
            request.form.get(
                "x_frac",
                "0.5"
            )
        )

    except (
        TypeError,
        ValueError
    ):

        return jsonify({
            "error":
            "Missing tap coordinates. "
            "Tap top and bottom of the stack first."
        }), 400

    if not (
        0 <= top_frac <
        bottom_frac <= 1
    ):

        return jsonify({
            "error":
            "Invalid tap positions."
        }), 400

    if not (
        0 <= x_frac <= 1
    ):

        return jsonify({
            "error":
            "Invalid horizontal tap position."
        }), 400

    filename = (
        f"ring_{os.getpid()}_"
        f"{int(time.time() * 1000)}"
        f"{ext}"
    )

    path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    try:

        file.save(
            path
        )

        result = process_image(
            path,
            top_frac,
            bottom_frac,
            x_frac
        )

        return jsonify(
            result
        )

    except Exception as e:

        logger.exception(
            "Ring counting failed"
        )

        return jsonify({
            "error": str(e)
        }), 500

    finally:

        if os.path.exists(path):

            try:
                os.remove(path)
            except OSError:
                pass


# ============================================================
# ERRORS
# ============================================================

@app.errorhandler(413)
def too_large(_):

    return jsonify({
        "error":
        "Photo too large. Maximum size is 10 MB."
    }), 413


@app.route("/status")
def status():

    return jsonify({
        "status": "ok",
        "version": "5.0-opencv-numpy"
    })


# ============================================================
# LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )