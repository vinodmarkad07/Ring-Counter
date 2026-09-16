import argparse
import os
import sys
import time

import cv2
import numpy as np
from scipy.signal import find_peaks

# ultralytics is optional -- the script must still run (manual --roi / auto
# Otsu fallback) on a machine where it isn't installed.
try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False


# --------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------
class Config:
    MAX_WIDTH = 1000
    CLAHE_CLIP = 3.0
    CLAHE_GRID = (8, 8)
    STRIP_WIDTH_FRAC = 0.12
    LOCAL_NORM_WINDOW = 41
    PEAK_DISTANCE_FRAC = 0.010
    PEAK_PROMINENCE_FRAC = 0.08
    GAP_OUTLIER_TOLERANCE = 0.55
    AUTO_DETECT_MIN_AREA_FRAC = 0.05
    AUTO_DETECT_MAX_AREA_FRAC = 0.85
    BLUR_WARN_THRESHOLD = 80
    BRIGHTNESS_WARN_LOW = 40
    BRIGHTNESS_WARN_HIGH = 220
    GLARE_WARN_FRAC = 0.02
    YOLO_CONF_THRESHOLD = 0.35     # min detection confidence to trust the YOLO box
    YOLO_MARGIN_FRAC = 0.02        # small margin added around the YOLO box


# --------------------------------------------------------------------------
# 2. IMAGE LOADING / PREPROCESSING
# --------------------------------------------------------------------------
def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_image(img: np.ndarray, max_width: int = Config.MAX_WIDTH) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    return cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE -- local contrast enhancement so faint ring seams stand out."""
    clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP, tileGridSize=Config.CLAHE_GRID)
    return clahe.apply(gray)


def check_blur(gray: np.ndarray) -> float:
    """Variance of the Laplacian; low value = blurry photo."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_glare(gray: np.ndarray, stack_box, overexposed_thresh: int = 240) -> float:
    """Fraction of overexposed pixels inside the stack box. High values mean
    a glare band is likely cutting through ring boundaries there."""
    x, y, w, h = stack_box
    region = gray[y:y + h, x:x + w]
    return float(np.mean(region > overexposed_thresh))


def check_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


# --------------------------------------------------------------------------
# 3. STACK REGION -- MANUAL > YOLO > AUTO (Otsu, last resort)
# --------------------------------------------------------------------------
_yolo_model_cache = {}


def load_yolo_model(model_path: str):
    """Cached loader so repeated calls (batch mode) don't reload weights."""
    if not _HAS_ULTRALYTICS:
        raise RuntimeError(
            "ultralytics is not installed. Run: pip install ultralytics"
        )
    if model_path not in _yolo_model_cache:
        _yolo_model_cache[model_path] = YOLO(model_path)
    return _yolo_model_cache[model_path]


def detect_stack_yolo(img_bgr: np.ndarray, model_path: str,
                       conf_thresh: float = Config.YOLO_CONF_THRESHOLD):
    """
    Runs a YOLO detector (trained on a single 'stack' class -- see
    train_stack_detector.py) on the resized image and returns the highest
    confidence box.

    This replaces Otsu/contour auto-detection as the primary fallback:
    Otsu grabs the whole busy factory background on cluttered scenes, which
    silently feeds the peak-counter thousands of irrelevant background rows.
    A model trained specifically to localize "the stack column" doesn't have
    that failure mode.

    Returns (stack_box, confidence, found). stack_box is None if not found.
    """
    model = load_yolo_model(model_path)
    h, w = img_bgr.shape[:2]

    results = model.predict(img_bgr, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None, 0.0, False

    confs = results.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    best_conf = float(confs[best_idx])

    if best_conf < conf_thresh:
        return None, best_conf, False

    xyxy = results.boxes.xyxy.cpu().numpy()[best_idx]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1

    margin_x = int(bw * Config.YOLO_MARGIN_FRAC)
    margin_y = int(bh * Config.YOLO_MARGIN_FRAC)
    x = max(0, int(x1) - margin_x)
    y = max(0, int(y1) - margin_y)
    bw = min(w - x, int(bw) + 2 * margin_x)
    bh = min(h - y, int(bh) + 2 * margin_y)

    return (x, y, bw, bh), best_conf, True


def detect_stack_auto(gray: np.ndarray):
    """
    Otsu threshold + largest contour. LAST-RESORT fallback only -- kept for
    machines with no YOLO model available. Works when the stack has a clean
    tonal separation from its background; on busy factory backgrounds it can
    fail (grab nearly the whole frame), which is exactly what was happening
    before the YOLO stage was added. The caller checks area_frac against
    Config.AUTO_DETECT_MIN/MAX_AREA_FRAC and treats an out-of-range result
    as unreliable.
    """
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


def stack_box_from_roi(gray: np.ndarray, roi_px: tuple[int, int, int, int]):
    h, w = gray.shape[:2]
    x, y, bw, bh = roi_px
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))
    return (x, y, bw, bh)


def extract_profile(gray: np.ndarray, stack_box):
    """
    Vertical strip through the centre of the stack box, collapsed into a
    1-D intensity profile by averaging across the strip's width (this
    suppresses per-column noise).
    """
    x, y, w, h = stack_box
    strip_w = max(3, int(w * Config.STRIP_WIDTH_FRAC))
    cx = x + w // 2
    x0 = max(x, cx - strip_w // 2)
    x1 = min(x + w, x0 + strip_w)
    strip = gray[y:y + h, x0:x1]
    profile = np.mean(strip.astype(np.float64), axis=1)
    return profile, (x0, y, x1 - x0, h)


def local_contrast_normalize(signal: np.ndarray, window: int) -> np.ndarray:
    """
    Subtracts a local moving average and divides by local std-dev.
    Suppresses broad glare gradients on polished rings while preserving
    the sharp local transitions that mark real ring boundaries.
    """
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


# --------------------------------------------------------------------------
# 4. GRADIENT + PEAK DETECTION
# --------------------------------------------------------------------------
def compute_gradient(profile: np.ndarray) -> np.ndarray:
    return np.abs(np.gradient(profile))


def find_ring_peaks(gradient: np.ndarray):
    length = len(gradient)
    if length < 5:
        return np.array([], dtype=int)
    rng = gradient.max() - gradient.min()
    prominence = max(rng * Config.PEAK_PROMINENCE_FRAC, 1e-6)
    distance = max(1, int(length * Config.PEAK_DISTANCE_FRAC))
    peaks, _ = find_peaks(gradient, prominence=prominence, distance=distance)
    return peaks


# --------------------------------------------------------------------------
# 5. GAP VALIDATION + COUNTING
# --------------------------------------------------------------------------
def validate_gaps(peaks: np.ndarray):
    """
    Drops peaks whose spacing to a neighbour deviates wildly from the
    median spacing (usually noise, not a real ring boundary). Returns the
    filtered peaks and a 0-1 spacing-consistency score.
    """
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


def count_rings(filtered_peaks: np.ndarray) -> int:
    return max(0, len(filtered_peaks) - 1) if len(filtered_peaks) >= 2 else len(filtered_peaks)


def compute_confidence(consistency: float, blur_score: float, brightness: float,
                        n_peaks: int, box_reliable: bool) -> str:
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


# --------------------------------------------------------------------------
# 6. DRAWING / OUTPUT
# --------------------------------------------------------------------------
def draw_result(img, stack_box, strip_box, peaks, ring_count, box_source):
    out = img.copy()
    sx, sy, sw, sh = stack_box
    rx, ry, rw, rh = strip_box

    box_color = {"manual": (255, 0, 0), "yolo": (0, 255, 0), "auto": (0, 165, 255)}
    cv2.rectangle(out, (sx, sy), (sx + sw, sy + sh), box_color.get(box_source, (0, 165, 255)), 2)
    cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 1)

    for p in peaks:
        y = ry + int(p)
        cv2.line(out, (sx, y), (sx + sw, y), (0, 255, 0), 1)

    cv2.putText(
        out, str(ring_count), (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        out, f"box: {box_source}", (10, 140),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color.get(box_source, (0, 165, 255)), 2, cv2.LINE_AA,
    )
    return out


# --------------------------------------------------------------------------
# 7. MAIN PIPELINE
# --------------------------------------------------------------------------
def process_image(path: str, roi_px=None, yolo_model_path: str = None, out_dir: str = "."):
    start = time.time()

    img = load_image(path)
    img = resize_image(img)
    gray = to_grayscale(img)

    blur_score = check_blur(gray)
    brightness = check_brightness(gray)
    enhanced = enhance_contrast(gray)

    box_source = None
    yolo_conf = None
    area_frac = None

    if roi_px is not None:
        # 1. Manual ROI always wins -- the operator/annotator knows best.
        stack_box = stack_box_from_roi(gray, roi_px)
        box_reliable = True
        box_source = "manual"

    else:
        stack_box = None
        if yolo_model_path:
            # 2. YOLO stack detector -- trained specifically to localize the
            #    stack column, so it doesn't grab the busy factory background
            #    the way Otsu thresholding does.
            try:
                yolo_box, yolo_conf, found = detect_stack_yolo(img, yolo_model_path)
                if found:
                    stack_box = yolo_box
                    box_reliable = True
                    box_source = "yolo"
                else:
                    print(f"WARNING: YOLO stack detector found nothing above "
                          f"confidence {Config.YOLO_CONF_THRESHOLD} (best={yolo_conf:.2f}). "
                          "Falling back to auto-detect.")
            except RuntimeError as e:
                print(f"WARNING: {e}. Falling back to auto-detect.")

        if stack_box is None:
            # 3. Last resort: Otsu + contour. Flagged unreliable on cluttered
            #    backgrounds -- this is the path that used to silently
            #    produce wildly inflated counts.
            stack_box, area_frac, box_reliable = detect_stack_auto(enhanced)
            box_source = "auto"
            if not box_reliable:
                print("WARNING: automatic stack detection looks unreliable "
                      f"(selected area covers {area_frac:.0%} of the frame). "
                      "Train/use a YOLO stack detector or pass --roi x,y,w,h "
                      "for an accurate count. See the saved *_resized_preview.jpg "
                      "to measure the box.")
                base = os.path.splitext(os.path.basename(path))[0]
                cv2.imwrite(os.path.join(out_dir, f"{base}_resized_preview.jpg"), img)

    profile, strip_box = extract_profile(enhanced, stack_box)
    glare_frac = check_glare(gray, stack_box)
    normalized = local_contrast_normalize(profile, Config.LOCAL_NORM_WINDOW)
    gradient = compute_gradient(normalized)
    peaks = find_ring_peaks(gradient)
    filtered_peaks, consistency = validate_gaps(peaks)

    ring_count = count_rings(filtered_peaks)
    confidence = compute_confidence(consistency, blur_score, brightness,
                                     len(filtered_peaks), box_reliable)
    elapsed = time.time() - start

    annotated = draw_result(img, stack_box, strip_box, filtered_peaks, ring_count, box_source)
    base = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(out_dir, f"{base}_output.jpg")
    crop_path = os.path.join(out_dir, f"{base}_crop.jpg")
    cv2.imwrite(output_path, annotated)

    sx, sy, sw, sh = stack_box
    cv2.imwrite(crop_path, annotated[sy:sy + sh, sx:sx + sw])

    print("-" * 40)
    print(f"Detected Rings   : {ring_count}")
    print(f"Confidence       : {confidence}")
    print(f"Gap Consistency  : {consistency:.2f} (1.0 = perfectly even spacing)")
    print(f"Processing Time  : {elapsed:.2f} sec")
    print(f"Blur Score       : {blur_score:.1f} (higher = sharper)")
    print(f"Brightness       : {brightness:.1f} (0-255)")
    print(f"Glare Fraction   : {glare_frac:.1%} of the stack box is overexposed")
    if confidence == "Low":
        print("NOTE: Low confidence -- treat this count as unreliable. "
              "Retake the photo following the capture checklist (full stack "
              "in frame, roughly straight-on, minimal glare) rather than "
              "trusting this number.")
    if box_source == "manual":
        print(f"Stack box source : manual --roi {roi_px}")
    elif box_source == "yolo":
        print(f"Stack box source : YOLO detector (confidence={yolo_conf:.2f})")
    else:
        print(f"Stack box source : auto-detected (reliable={box_reliable}, "
              f"area_frac={area_frac:.2f})")
    print(f"Saved Annotated Image -> {output_path}")
    print(f"Saved Bounding-Box Crop -> {crop_path}")
    print("-" * 40)

    return {
        "ring_count": ring_count,
        "confidence": confidence,
        "gap_consistency": consistency,
        "box_source": box_source,
        "elapsed": elapsed,
        "output_path": output_path,
        "crop_path": crop_path,
    }


# --------------------------------------------------------------------------
# 8. CLI ENTRY POINT
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Count rings in a photographed stack")
    parser.add_argument("image", nargs="?", default=None, help="path to the stack photo")
    parser.add_argument("--roi", type=str, default=None,
                         help="x,y,w,h in pixels, measured on the image AFTER it is "
                              "resized to width %d. Takes priority over --yolo-model "
                              "and auto-detect." % Config.MAX_WIDTH)
    parser.add_argument("--yolo-model", type=str, default=None,
                         help="path to a trained YOLO stack-detector .pt file "
                              "(see train_stack_detector.py). Used when --roi is "
                              "not given. Strongly recommended over relying on "
                              "auto-detect for cluttered factory backgrounds.")
    parser.add_argument("--out-dir", default=".", help="where to save output images")
    args = parser.parse_args()

    if not args.image:
        print("Usage: python counter.py path/to/image.jpg [--roi x,y,w,h] [--yolo-model best.pt]")
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"Error: image file not found -> {args.image}")
        sys.exit(1)

    roi_px = None
    if args.roi:
        try:
            roi_px = tuple(int(v.strip()) for v in args.roi.split(","))
            if len(roi_px) != 4:
                raise ValueError
        except ValueError:
            print("Error: --roi must be four comma-separated integers, e.g. --roi 350,20,350,960")
            sys.exit(1)

    process_image(args.image, roi_px=roi_px, yolo_model_path=args.yolo_model, out_dir=args.out_dir)


if __name__ == "__main__":
    main()