"""
counter.py v2 -- Manual-Axis Edition
============================================================
Why this version exists:
  Testing v1 on a real client photo (ring-full-1789643106930.jpg)
  found detect_stack_auto() (Otsu threshold + largest contour) grab
  the ENTIRE frame -- area_frac=1.0, reliable=False -- because the
  factory background didn't separate cleanly from the stack. The
  pipeline still ran the peak counter on that bad box and reported
  a confident-looking number (48) mixing real ring seams with
  background clutter. That's not a tuning problem, it's a wrong
  input region, same root cause as the earlier web-app bug.

  Two changes fix this:
  1. Manual --top-y/--bottom-y (or interactive click-to-mark) is now
     the PRIMARY path, not an equal-priority option. It guarantees
     the measuring axis sits inside the real stack because a human
     confirms it, which is what actually drives accuracy -- YOLO
     and Otsu are still available but demoted to explicit opt-in
     fallbacks for batch/unattended use.
  2. When no reliable box exists (Otsu fails its own area-fraction
     check, or YOLO confidence is below threshold), the script now
     REFUSES to print a ring_count. It reports what went wrong and
     exits non-zero instead of silently handing back a number built
     on a bad region. A confident-looking wrong number is worse
     than no number.

  This does not claim a fixed accuracy percentage. No ground-truth
  comparison exists in this codebase to support one. Confidence
  labels (High/Medium/Low) are relative signal-quality indicators,
  not measured accuracy -- treat "Low" as "recount by hand."
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
from scipy.signal import find_peaks

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False


# --------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------
class Config:
    MAX_WIDTH = 1200
    CLAHE_CLIP = 2.5
    CLAHE_GRID = (8, 8)
    N_STRIPS = 9                    # multi-strip voting (ported from the web app fix)
    STRIP_WIDTH_FRAC = 0.06
    AXIS_HALF_WIDTH_FRAC = 0.22
    LOCAL_NORM_WINDOW = 41
    PEAK_DISTANCE_FRAC = 0.015
    PEAK_PROMINENCE_FRAC = 0.09
    VOTE_PROMINENCE_FRAC = 0.10
    GAP_OUTLIER_TOLERANCE = 0.35
    MIN_STRIPS_AGREEING = 0.55      # fraction of strips that must agree on a peak
    AUTO_DETECT_MIN_AREA_FRAC = 0.05
    AUTO_DETECT_MAX_AREA_FRAC = 0.85
    BLUR_WARN_THRESHOLD = 60
    BRIGHTNESS_WARN_LOW = 35
    BRIGHTNESS_WARN_HIGH = 225
    GLARE_THRESH = 235
    YOLO_CONF_THRESHOLD = 0.35
    YOLO_MARGIN_FRAC = 0.02


# --------------------------------------------------------------------------
# 2. IMAGE LOADING / PREPROCESSING
# --------------------------------------------------------------------------
def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_image(img: np.ndarray, max_width: int = Config.MAX_WIDTH):
    h, w = img.shape[:2]
    if w <= max_width:
        return img, 1.0
    scale = max_width / float(w)
    return cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA), scale


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP, tileGridSize=Config.CLAHE_GRID)
    return clahe.apply(gray)


def check_blur(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_glare(gray: np.ndarray, stack_box, overexposed_thresh: int = Config.GLARE_THRESH) -> float:
    x, y, w, h = stack_box
    region = gray[y:y + h, x:x + w]
    return float(np.mean(region > overexposed_thresh)) if region.size else 0.0


def check_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


# --------------------------------------------------------------------------
# 3. STACK REGION -- MANUAL AXIS (primary) > YOLO > AUTO (last resort, gated)
# --------------------------------------------------------------------------
def stack_box_from_manual_axis(gray_shape, top_y: int, bottom_y: int, center_x: int = None):
    """
    The primary path. top_y/bottom_y are pixel rows on the RESIZED image
    (width = Config.MAX_WIDTH) marking the top and bottom of the stack --
    get them by eye from the *_resized_preview.jpg this script saves, or
    via --interactive to click them directly.
    """
    h, w = gray_shape[:2]
    if bottom_y <= top_y + 20:
        raise ValueError("bottom-y must be clearly below top-y (got "
                          f"top={top_y}, bottom={bottom_y}).")
    top_y = max(0, min(top_y, h - 1))
    bottom_y = max(top_y + 20, min(bottom_y, h))
    cx = center_x if center_x is not None else w // 2
    axis_h = bottom_y - top_y
    axis_w = max(20, int(axis_h * Config.AXIS_HALF_WIDTH_FRAC * 2))
    axis_x = max(0, cx - axis_w // 2)
    axis_w = min(axis_w, w - axis_x)
    return (axis_x, top_y, axis_w, axis_h)


def stack_box_from_roi(gray: np.ndarray, roi_px):
    h, w = gray.shape[:2]
    x, y, bw, bh = roi_px
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))
    return (x, y, bw, bh)


_yolo_model_cache = {}


def load_yolo_model(model_path: str):
    if not _HAS_ULTRALYTICS:
        raise RuntimeError("ultralytics is not installed. Run: pip install ultralytics")
    if model_path not in _yolo_model_cache:
        _yolo_model_cache[model_path] = YOLO(model_path)
    return _yolo_model_cache[model_path]


def detect_stack_yolo(img_bgr: np.ndarray, model_path: str,
                       conf_thresh: float = Config.YOLO_CONF_THRESHOLD):
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
    LAST-RESORT fallback only. On cluttered backgrounds this can grab
    nearly the whole frame -- that is exactly what happened during
    testing (area_frac=1.0, reliable=False on a real client photo).
    The caller MUST treat an unreliable result as "no box", not as a
    degraded-but-usable one -- see the hard gate in process_image().
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


# --------------------------------------------------------------------------
# 4. MULTI-STRIP VOTING PEAK DETECTION (ported from the web-app fix --
#    tested to reject background clutter better than a single center strip)
# --------------------------------------------------------------------------
def suppress_glare_rows(strip, thresh=Config.GLARE_THRESH):
    col_means = np.mean(strip, axis=1)
    glare_rows = col_means > thresh
    if glare_rows.sum() > 0:
        good_mean = float(np.median(col_means[~glare_rows])) if (~glare_rows).sum() > 0 else 128.0
        strip = strip.copy()
        strip[glare_rows, :] = int(good_mean)
    return strip


def local_contrast_normalize(signal: np.ndarray, window: int) -> np.ndarray:
    window = max(3, window | 1)
    pad = window // 2
    padded = np.pad(signal, pad, mode="reflect")
    kernel = np.ones(window) / window
    local_mean = np.convolve(padded, kernel, mode="valid")
    local_sq_mean = np.convolve(padded ** 2, kernel, mode="valid")
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 1e-6))
    return (signal - local_mean) / local_std


def get_peaks_from_profile(profile, n_rows):
    normalized = local_contrast_normalize(profile, Config.LOCAL_NORM_WINDOW)
    gradient = np.abs(np.gradient(normalized))
    rng = gradient.max() - gradient.min()
    prominence = max(rng * Config.PEAK_PROMINENCE_FRAC, 1e-6)
    distance = max(1, int(n_rows * Config.PEAK_DISTANCE_FRAC))
    peaks, _ = find_peaks(gradient, prominence=prominence, distance=distance)
    return peaks


def multi_strip_vote(enhanced, stack_box):
    sx, sy, sw, sh = stack_box
    n = Config.N_STRIPS
    sw_strip = max(3, int(sw * Config.STRIP_WIDTH_FRAC))

    margin = int(sw * 0.08)
    xs = np.linspace(sx + margin, sx + sw - margin, n, dtype=int)

    vote_map = np.zeros(sh, dtype=float)
    strips_hit = np.zeros(sh, dtype=int)

    for cx in xs:
        x0 = max(sx, cx - sw_strip // 2)
        x1 = min(sx + sw, x0 + sw_strip)
        strip = enhanced[sy:sy + sh, x0:x1]
        strip = suppress_glare_rows(strip)
        profile = np.mean(strip.astype(np.float64), axis=1)
        peaks = get_peaks_from_profile(profile, sh)
        sigma = max(2, sh * 0.007)
        rr = np.arange(sh)
        for p in peaks:
            vote_map += np.exp(-0.5 * ((rr - p) / sigma) ** 2)
            strips_hit[max(0, p - 2):min(sh, p + 3)] += 1

    vote_map /= n

    if vote_map.max() < 1e-6:
        return np.array([], dtype=int), 0

    vmin, vmax = vote_map.min(), vote_map.max()
    vote_norm = (vote_map - vmin) / (vmax - vmin + 1e-9)

    rng = vote_norm.max() - vote_norm.min()
    prominence = max(rng * Config.VOTE_PROMINENCE_FRAC, 0.03)
    distance = max(1, int(sh * Config.PEAK_DISTANCE_FRAC))

    candidate_peaks, _ = find_peaks(vote_norm, prominence=prominence, distance=distance)

    min_hits = max(1, int(np.ceil(n * Config.MIN_STRIPS_AGREEING)))
    confirmed = np.array([p for p in candidate_peaks if strips_hit[p] >= min_hits], dtype=int)

    rejected_count = len(candidate_peaks) - len(confirmed)
    return confirmed, rejected_count


# --------------------------------------------------------------------------
# 5. GAP VALIDATION + COUNTING
# --------------------------------------------------------------------------
def validate_gaps(peaks: np.ndarray):
    if len(peaks) < 3:
        return peaks, 0.0
    gaps = np.diff(peaks)
    median_gap = np.median(gaps)
    if median_gap <= 0:
        return peaks, 0.0
    keep = [True] + [abs(g - median_gap) / median_gap <= Config.GAP_OUTLIER_TOLERANCE for g in gaps]
    filtered = peaks[np.array(keep)]
    gaps_after = np.diff(filtered) if len(filtered) > 1 else np.array([median_gap])
    consistency = 1.0 - min(1.0, float(np.std(gaps_after) / (np.mean(gaps_after) + 1e-9)))
    return filtered, max(0.0, consistency)


def count_rings(filtered_peaks: np.ndarray) -> int:
    return max(0, len(filtered_peaks) - 1) if len(filtered_peaks) >= 2 else len(filtered_peaks)


def compute_confidence(consistency: float, blur_score: float, brightness: float,
                        n_peaks: int, rejected_count: int) -> str:
    score = consistency
    if blur_score < Config.BLUR_WARN_THRESHOLD:
        score -= 0.25
    if brightness < Config.BRIGHTNESS_WARN_LOW or brightness > Config.BRIGHTNESS_WARN_HIGH:
        score -= 0.15
    if n_peaks < 4:
        score -= 0.25
    if rejected_count > n_peaks * 0.3:
        score -= 0.15
    if score >= 0.65:
        return "High"
    elif score >= 0.35:
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------
# 6. DRAWING / OUTPUT
# --------------------------------------------------------------------------
def draw_result(img, stack_box, peaks, ring_count, box_source):
    out = img.copy()
    sx, sy, sw, sh = stack_box

    box_color = {"manual-axis": (0, 255, 0), "manual-roi": (255, 0, 0),
                 "yolo": (0, 200, 200), "auto": (0, 165, 255)}
    cv2.rectangle(out, (sx, sy), (sx + sw, sy + sh), box_color.get(box_source, (0, 165, 255)), 2)

    for p in peaks:
        y = sy + int(p)
        cv2.line(out, (sx, y), (sx + sw, y), (0, 255, 80), 2)

    cv2.putText(out, str(ring_count), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 7, cv2.LINE_AA)
    cv2.putText(out, str(ring_count), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, f"box: {box_source}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color.get(box_source, (0, 165, 255)), 2, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# 7. INTERACTIVE AXIS PICKER (click top, click bottom, press any key)
# --------------------------------------------------------------------------
def pick_axis_interactively(img):
    """Opens an OpenCV window; click the top of the stack, then the bottom.
    Requires a display (won't work over a headless SSH session)."""
    points = []
    win = "Click TOP of stack, then BOTTOM -- any key to confirm"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    disp = img.copy()
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    while True:
        frame = disp.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(frame, (x, y), 6, (0, 255, 0) if i == 0 else (255, 0, 0), -1)
            cv2.line(frame, (0, y), (frame.shape[1], y), (0, 255, 0) if i == 0 else (255, 0, 0), 2)
        cv2.imshow(win, frame)
        key = cv2.waitKey(30)
        if len(points) == 2 or key != -1:
            break
    cv2.destroyWindow(win)
    if len(points) < 2:
        raise ValueError("Need two clicks (top and bottom of stack) to proceed.")
    (x0, y0), (x1, y1) = points
    top_y, bottom_y = sorted([y0, y1])
    center_x = (x0 + x1) // 2
    return top_y, bottom_y, center_x


# --------------------------------------------------------------------------
# 8. MAIN PIPELINE
# --------------------------------------------------------------------------
def process_image(path: str, top_y=None, bottom_y=None, center_x=None,
                   roi_px=None, yolo_model_path=None, interactive=False,
                   out_dir: str = "."):
    start = time.time()

    img = load_image(path)
    img, scale = resize_image(img)
    gray = to_grayscale(img)
    enhanced = enhance_contrast(gray)

    blur_score = check_blur(gray)
    brightness = check_brightness(gray)

    box_source = None
    yolo_conf = None
    area_frac = None
    box_reliable = False
    stack_box = None

    base = os.path.splitext(os.path.basename(path))[0]
    preview_path = os.path.join(out_dir, f"{base}_resized_preview.jpg")

    if interactive:
        top_y, bottom_y, center_x = pick_axis_interactively(img)

    if top_y is not None and bottom_y is not None:
        # 1. Manual axis -- PRIMARY path, guaranteed inside the real stack.
        stack_box = stack_box_from_manual_axis(gray.shape, top_y, bottom_y, center_x)
        box_reliable = True
        box_source = "manual-axis"

    elif roi_px is not None:
        stack_box = stack_box_from_roi(gray, roi_px)
        box_reliable = True
        box_source = "manual-roi"

    else:
        if yolo_model_path:
            try:
                yolo_box, yolo_conf, found = detect_stack_yolo(img, yolo_model_path)
                if found:
                    stack_box = yolo_box
                    box_reliable = True
                    box_source = "yolo"
                else:
                    print(f"WARNING: YOLO stack detector found nothing above "
                          f"confidence {Config.YOLO_CONF_THRESHOLD} (best={yolo_conf:.2f}).")
            except RuntimeError as e:
                print(f"WARNING: {e}")

        if stack_box is None:
            stack_box, area_frac, box_reliable = detect_stack_auto(enhanced)
            box_source = "auto"

        # HARD GATE -- an unreliable box no longer produces a count.
        if not box_reliable:
            cv2.imwrite(preview_path, img)
            print("-" * 60)
            print("REFUSING TO COUNT: no reliable stack region found.")
            if box_source == "auto":
                print(f"  Auto-detect (Otsu) selected {area_frac:.0%} of the frame, "
                      "which is outside the trusted range -- it likely grabbed "
                      "background clutter, not just the stack.")
            print(f"  Saved preview -> {preview_path}")
            print("  Fix: re-run with --top-y/--bottom-y (measure on the preview "
                  "image above), or --interactive to click them, or --roi, or a "
                  "trained --yolo-model.")
            print("-" * 60)
            return {
                "ring_count": None,
                "confidence": "REFUSED",
                "box_source": box_source,
                "reason": "unreliable_stack_region",
                "preview_path": preview_path,
            }

    glare_frac = check_glare(gray, stack_box)
    confirmed_peaks, rejected_count = multi_strip_vote(enhanced, stack_box)
    filtered_peaks, consistency = validate_gaps(confirmed_peaks)

    ring_count = count_rings(filtered_peaks)
    confidence = compute_confidence(consistency, blur_score, brightness,
                                     len(filtered_peaks), rejected_count)
    elapsed = time.time() - start

    annotated = draw_result(img, stack_box, filtered_peaks, ring_count, box_source)
    output_path = os.path.join(out_dir, f"{base}_output.jpg")
    crop_path = os.path.join(out_dir, f"{base}_crop.jpg")
    cv2.imwrite(output_path, annotated)

    sx, sy, sw, sh = stack_box
    cv2.imwrite(crop_path, annotated[sy:sy + sh, sx:sx + sw])

    print("-" * 60)
    print(f"Detected Rings   : {ring_count}")
    print(f"Confidence       : {confidence}")
    print(f"Gap Consistency  : {consistency:.2f} (1.0 = perfectly even spacing)")
    print(f"Rejected peaks   : {rejected_count} (candidates too few strips agreed on)")
    print(f"Processing Time  : {elapsed:.2f} sec")
    print(f"Blur Score       : {blur_score:.1f} (higher = sharper)")
    print(f"Brightness       : {brightness:.1f} (0-255)")
    print(f"Glare Fraction   : {glare_frac:.1%} of the stack box is overexposed")
    print(f"Stack box source : {box_source}")
    if confidence == "Low":
        print("NOTE: Low confidence -- treat this count as unreliable. Verify "
              f"by eye against {crop_path} or recount by hand.")
    print(f"Saved Annotated Image -> {output_path}")
    print(f"Saved Bounding-Box Crop -> {crop_path}")
    print("-" * 60)

    return {
        "ring_count": ring_count,
        "confidence": confidence,
        "gap_consistency": consistency,
        "rejected_peaks": rejected_count,
        "box_source": box_source,
        "elapsed": elapsed,
        "output_path": output_path,
        "crop_path": crop_path,
    }


# --------------------------------------------------------------------------
# 9. CLI ENTRY POINT
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Count rings in a photographed stack")
    parser.add_argument("image", nargs="?", default=None, help="path to the stack photo")
    parser.add_argument("--top-y", type=int, default=None,
                         help="pixel row of the TOP of the stack, measured on the image "
                              "after it's resized to width %d. Run once without this flag "
                              "to get a *_resized_preview.jpg to measure on." % Config.MAX_WIDTH)
    parser.add_argument("--bottom-y", type=int, default=None,
                         help="pixel row of the BOTTOM of the stack (same resized coordinates).")
    parser.add_argument("--center-x", type=int, default=None,
                         help="optional pixel column of the stack's horizontal center; "
                              "defaults to image center.")
    parser.add_argument("--interactive", action="store_true",
                         help="open a window to click the top and bottom of the stack "
                              "instead of passing --top-y/--bottom-y. Needs a display.")
    parser.add_argument("--roi", type=str, default=None,
                         help="x,y,w,h in pixels (resized coordinates). Alternative to "
                              "--top-y/--bottom-y for a full box instead of an axis.")
    parser.add_argument("--yolo-model", type=str, default=None,
                         help="path to a trained YOLO stack-detector .pt file. Used only "
                              "when no manual axis/ROI is given.")
    parser.add_argument("--out-dir", default=".", help="where to save output images")
    args = parser.parse_args()

    if not args.image:
        print("Usage:")
        print("  python counter.py photo.jpg --top-y 40 --bottom-y 1200")
        print("  python counter.py photo.jpg --interactive")
        print("  python counter.py photo.jpg   (no axis -> saves a preview to measure on)")
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

    result = process_image(
        args.image,
        top_y=args.top_y, bottom_y=args.bottom_y, center_x=args.center_x,
        roi_px=roi_px, yolo_model_path=args.yolo_model,
        interactive=args.interactive, out_dir=args.out_dir,
    )

    if result["ring_count"] is None:
        sys.exit(2)  # refused -- caller/script can detect this exit code


if __name__ == "__main__":
    main()
