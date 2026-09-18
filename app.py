"""RingCount AI Flask application.

Run locally with:
    python app.py

The CV engine is loaded once at process startup.  Requests pass image bytes
directly to the engine, so temporary uploads are not retained on disk.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import SETTINGS
from core.detector import RingCounter, RingDetectionError
from core.storage import HistoryStore


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ringcount")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = SETTINGS.MAX_CONTENT_LENGTH
app.config["JSON_SORT_KEYS"] = False

counter = RingCounter()
history = HistoryStore(SETTINGS.DATABASE_PATH, SETTINGS.HISTORY_LIMIT)


def _error(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status


def _float_field(name: str, default: float | None = None) -> float | None:
    raw = request.form.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise RingDetectionError(f"{name} must be a number.")


@app.get("/")
def index():
    return render_template(
        "index.html",
        version=SETTINGS.APP_VERSION,
        max_upload_mb=SETTINGS.MAX_UPLOAD_MB,
    )


@app.get("/status")
@app.get("/api/health")
def status():
    return jsonify(
        {
            "success": True,
            "status": "ok",
            "version": SETTINGS.APP_VERSION,
            "model": {
                "loaded": counter.model_loaded,
                "path": str(SETTINGS.MODEL_PATH),
                "error": counter.model_error,
            },
            "device": counter.device,
            "cuda": counter.device == "cuda",
            "engine": "hybrid-cv-with-optional-yolo",
        }
    )


@app.post("/count")
@app.post("/api/detect")
def count():
    upload = request.files.get("photo") or request.files.get("image")
    if upload is None:
        return _error("No image was uploaded. Choose a JPG, PNG, or WEBP photo.")
    if not upload.filename:
        return _error("The selected file has no name.")

    extension = Path(upload.filename).suffix.lower()
    if extension not in SETTINGS.ALLOWED_EXTENSIONS:
        return _error("Unsupported file type. Use JPG, JPEG, PNG, or WEBP.")

    auto_roi = request.form.get("auto_roi", "0").lower() in {"1", "true", "yes"}
    try:
        top = _float_field("top_frac", 0.04 if auto_roi else None)
        bottom = _float_field("bottom_frac", 0.96 if auto_roi else None)
        x_fraction = _float_field("x_frac", 0.5)
        if top is None or bottom is None:
            raise RingDetectionError(
                "Tap the top and bottom of the stack before analyzing the image."
            )
        data = upload.read()
        if len(data) > SETTINGS.MAX_CONTENT_LENGTH:
            return _error(
                f"Photo too large. The maximum upload size is {SETTINGS.MAX_UPLOAD_MB:g} MB.",
                413,
            )
        result = counter.analyze_bytes(
            data, top, bottom, x_fraction, auto_roi=auto_roi
        )
        result["image_name"] = secure_filename(upload.filename) or "photo.jpg"
        history.add(result["image_name"], result)
        return jsonify(result)
    except RingDetectionError as exc:
        return _error(str(exc), 400)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception:
        logger.exception("Unexpected count failure")
        return _error("The image could not be processed. Try another photo.", 500)


@app.get("/api/history")
def get_history():
    try:
        limit = int(request.args.get("limit", 25))
    except ValueError:
        limit = 25
    return jsonify({"success": True, "items": history.list(limit)})


@app.delete("/api/history/<int:record_id>")
def delete_history(record_id: int):
    if not history.delete(record_id):
        return _error("History record not found.", 404)
    return jsonify({"success": True})


@app.get("/api/history.csv")
def export_history():
    payload = history.csv_bytes()
    from io import BytesIO

    return send_file(
        BytesIO(payload),
        mimetype="text/csv",
        as_attachment=True,
        download_name="ringcount-history.csv",
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    return _error(
        f"Photo too large. The maximum upload size is {SETTINGS.MAX_UPLOAD_MB:g} MB.",
        413,
    )


@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/") or request.path == "/count":
        return _error("Endpoint not found.", 404)
    return render_template("index.html", version=SETTINGS.APP_VERSION, max_upload_mb=SETTINGS.MAX_UPLOAD_MB), 404


@app.errorhandler(Exception)
def handle_unexpected(error):
    if isinstance(error, (BrokenPipeError, ConnectionError)):
        return _error("The connection ended before processing completed.", 500)
    logger.exception("Unhandled server error: %s", error)
    return _error("Unexpected server error.", 500)


if __name__ == "__main__":
    app.run(host=SETTINGS.HOST, port=SETTINGS.PORT, debug=SETTINGS.DEBUG)