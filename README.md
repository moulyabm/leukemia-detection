# ALL/AML Leukemia Detection System
**AI-powered blood smear analysis for detecting Acute Lymphoblastic (ALL) and Acute Myeloid (AML) Leukemia**

---

## Project Structure
```
leukemia_detection/
├── frontend/
│   └── index.html           ← Single-file frontend (HTML + CSS + JS)
├── backend/
│   ├── app.py               ← Flask REST API
│   └── requirements.txt     ← Python dependencies
├── ml_model/
│   ├── model.py             ← CNN model definition + LeukemiaDetector class
│   └── saved_model/         ← (created after training) weights + meta
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

To use the full TensorFlow CNN (GPU/CPU training + inference), also run:
```bash
pip install tensorflow   # or tensorflow-cpu for CPU-only
```

### 2. Run the server
```bash
cd backend
python app.py
```
Open **http://localhost:5000** in your browser.

> **Demo mode**: If TensorFlow or a trained model is not available, the backend
> still runs using the built-in mock inference engine. The frontend also has a
> full client-side demo mode that activates automatically when the backend is
> offline.

---

## Training Your Own Model

### Dataset
Use the [ALL-IDB dataset](https://homes.di.unimi.it/scotti/all/) or the
[Kaggle Leukemia Classification dataset](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification).

Organise data like this:
```
data/
  train/
    ALL/   ← ALL microscopy images
    AML/   ← AML microscopy images
  val/
    ALL/
    AML/
```

### Train
```bash
cd ml_model
python model.py train /path/to/data/train /path/to/data/val
```

The trained model is saved to `ml_model/saved_model/leukemia_cnn.h5`.

---

## API Reference

### `GET /api/health`
Returns server and model status.

### `GET /api/model-info`
Returns model metadata (version, class names, accuracy).

### `POST /api/analyze`
Analyze a blood smear image.

**Request:** `multipart/form-data` with field `image` (PNG/JPG/JPEG/BMP/TIFF)

**Response:**
```json
{
  "status": "ok",
  "result": {
    "step1": { "brightness": 62.3, "contrast": 78.5, "qualityScore": 95, ... },
    "step2": { "cancerCellsDetected": 28, "blastCellPercentage": 42.1, ... },
    "step3": { "morphological": {...}, "textural": {...}, "cellular": {...} },
    "step4": { "type": "ALL", "confidence": 94.2, "stage": "Stage II", ... },
    "step5": { "accuracy": 99.4, "precision": 98.1, "auc": 0.994, ... },
    "heatmap": { "hotspots": 7, "method": "Grad-CAM" },
    "processingTimeMs": 1234
  }
}
```

---

## ML Model Architecture

```
Input (224×224×3)
  → Stem (Conv 32→64, MaxPool)
  → ResBlock × 3 (64, 128, 256 filters)
  → Squeeze-Excitation Attention
  → GlobalAvgPool
  → Dense 512 → BatchNorm → Dropout(0.5)
  → Dense 256 → Dropout(0.3)
  → Softmax output (ALL | AML)
```

Key design choices:
- **Residual blocks** prevent vanishing gradients on deep networks
- **SE attention** lets the model focus on diagnostically relevant channels
- **Heavy augmentation** (rotation, flip, brightness) improves generalisation on small medical datasets

---

## Technology Stack

| Layer     | Technology                         |
|-----------|------------------------------------|
| Frontend  | Vanilla HTML5 / CSS3 / JavaScript  |
| Backend   | Python 3.10+ · Flask · Flask-CORS  |
| ML Model  | TensorFlow 2.x · Keras · NumPy     |
| Images    | Pillow                             |
| Deploy    | Gunicorn (production WSGI)         |

---

## Disclaimer

> This system is intended for **research and educational purposes only**.
> It is NOT a certified medical device and must not be used as a substitute
> for professional medical diagnosis. Always consult a qualified haematologist.
