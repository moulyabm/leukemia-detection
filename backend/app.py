"""
ALL/AML Leukemia Detection — Flask Backend API
===============================================
RESTful API that wraps the ML inference engine and serves the frontend.

Endpoints:
    POST /api/analyze          — Analyze a blood smear image
    GET  /api/health           — Health check
    GET  /api/model-info       — Model metadata
"""

import os
import sys
import uuid
import json
import time
import logging
from pathlib import Path
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "ml_model"))
from model import LeukemiaDetector

# ── App setup ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Configuration ──────────────────────────────────────────────────────────────
UPLOAD_FOLDER    = Path(__file__).parent / "uploads"
MAX_CONTENT_MB   = 16
ALLOWED_EXTS     = {"png", "jpg", "jpeg", "bmp", "tiff", "tif"}
MODEL_PATH       = Path(__file__).parent.parent / "ml_model" / "saved_model" / "leukemia_cnn.h5"

app.config["UPLOAD_FOLDER"]    = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# ── Load detector ──────────────────────────────────────────────────────────────
detector = LeukemiaDetector(model_path=str(MODEL_PATH))
logger.info("LeukemiaDetector initialised.")


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def api_response(data: dict, status: int = 200):
    return jsonify({"status": "ok" if status < 400 else "error", **data}), status


def require_json_or_form(f):
    """Decorator — no special check needed, just documents intent."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
#  Routes — Static Frontend
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ══════════════════════════════════════════════════════════════════════════════
#  Routes — API
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return api_response({
        "model_loaded": detector.model is not None,
        "upload_dir":   str(UPLOAD_FOLDER),
        "allowed_types": list(ALLOWED_EXTS),
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model metadata."""
    meta_path = Path(__file__).parent.parent / "ml_model" / "saved_model" / "model_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {
            "model_version": "1.0.0-demo",
            "img_size": 224,
            "class_names": ["ALL", "AML"],
            "note": "Demo mode — no trained weights found.",
        }
    return api_response({"model": meta})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analyze a blood smear image.

    Request: multipart/form-data  with field  "image"
    Response: full analysis JSON (5-step pipeline)
    """
    t_start = time.time()

    # ── Validate upload ────────────────────────────────────────────────────────
    if "image" not in request.files:
        return api_response({"message": "No image field in request."}, 400)

    file = request.files["image"]
    if file.filename == "":
        return api_response({"message": "Empty filename."}, 400)

    if not allowed_file(file.filename):
        return api_response({
            "message": f"File type not allowed. Accepted: {', '.join(ALLOWED_EXTS)}"
        }, 415)

    # ── Save to disk ───────────────────────────────────────────────────────────
    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    save_path   = UPLOAD_FOLDER / unique_name
    file.save(str(save_path))
    logger.info(f"Saved upload: {save_path}")

    try:
        # ── Run inference ──────────────────────────────────────────────────────
        prediction  = detector.predict(str(save_path))
        heatmap     = detector.generate_gradcam(str(save_path))
        elapsed_ms  = round((time.time() - t_start) * 1000)

        # ── Build 5-step result ────────────────────────────────────────────────
        result = _build_pipeline_result(prediction, heatmap)
        result["processingTimeMs"] = elapsed_ms
        result["filename"]         = file.filename

        logger.info(
            f"Analysis complete: {prediction['label']} "
            f"({prediction['confidence']}%) in {elapsed_ms}ms"
        )
        return api_response({"result": result})

    except Exception as e:
        logger.exception("Inference failed")
        return api_response({"message": f"Analysis failed: {str(e)}"}, 500)

    finally:
        # Clean up uploaded file
        try:
            os.remove(save_path)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline Result Builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_pipeline_result(pred: dict, heatmap: dict) -> dict:
    """Map raw model output to the 5-step pipeline format the frontend expects."""
    pre  = pred["preprocessing"]
    seg  = pred["segmentation"]
    feat = pred["features"]
    met  = pred["metrics"]

    return {
        # ── Step 1: Preprocessing ──────────────────────────────────────────────
        "step1": {
            "brightness":   pre["brightness"],
            "contrast":     pre["contrast"],
            "clarity":      pre["clarity"],
            "qualityScore": pre["qualityScore"],
            "resolution":   pre["resolution"],
            "colorSpace":   pre["colorSpace"],
        },

        # ── Step 2: Segmentation ───────────────────────────────────────────────
        "step2": {
            "segmentationMethod":  seg["segmentationMethod"],
            "cancerCellsDetected": seg["cancerCellsDetected"],
            "normalCellsDetected": seg["normalCellsDetected"],
            "totalCells":          seg["totalCells"],
            "blastCellPercentage": seg["blastCellPercentage"],
        },

        # ── Step 3: Feature Extraction ─────────────────────────────────────────
        "step3": {
            "morphological": feat["morphological"],
            "textural":      feat["textural"],
            "cellular":      feat["cellular"],
        },

        # ── Step 4: Classification ─────────────────────────────────────────────
        "step4": {
            "type":       pred["label"],
            "confidence": pred["confidence"],
            "stage":      pred["stage"],
            "severity":   pred["severity"],
            "algorithm":  "CNN + Squeeze-Excitation Attention",
            "modelVersion": "1.0.0",
            "comparisonScores": {
                "ALL": pred["all_score"],
                "AML": pred["aml_score"],
            },
        },

        # ── Step 5: Performance Metrics ────────────────────────────────────────
        "step5": {
            "accuracy":    met["accuracy"],
            "precision":   met["precision"],
            "recall":      met["recall"],
            "f1Score":     met["f1Score"],
            "auc":         met["auc"],
            "specificity": met["specificity"],
            "sensitivity": met["sensitivity"],
        },

        # ── Heatmap ────────────────────────────────────────────────────────────
        "heatmap": heatmap,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Error Handlers
# ══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return api_response(
        {"message": f"File too large. Maximum size: {MAX_CONTENT_MB} MB."}, 413
    )


@app.errorhandler(404)
def not_found(_):
    return api_response({"message": "Endpoint not found."}, 404)


@app.errorhandler(500)
def internal_error(_):
    return api_response({"message": "Internal server error."}, 500)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting server on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
