# RingCount AI

RingCount AI counts cylindrical metal rings in a vertically stacked pile from
a camera, gallery, or desktop image. The operator uploads a compressed image,
marks the visible stack top and bottom, and reviews an annotated result.

The project keeps the reliable part of the original v4 workflow — manual ROI
selection — and separates it from the Flask API and storage layer. The
detector uses:

- CLAHE/local contrast preprocessing
- robust row-wise edge and intensity profiles
- multiple horizontal strips and cross-strip voting
- duplicate suppression and spacing statistics
- controlled missing-gap hypotheses
- optional YOLO detection fusion when `weights/ring_best.pt` and Ultralytics
  are installed
- CPU fallback when no model or CUDA device is available

**Important:** confidence is evidence quality, not accuracy. The application
does not hard-code the known answers 44, 47, or 37 and does not claim 90%
accuracy until `evaluate.py` measures it on a separated, labelled test set.

## Project layout

```text
app.py                         Flask API and startup wiring
config.py                      Environment-backed configuration
core/detector.py               Reusable hybrid counting engine
core/preprocessing.py          Decode, resize, quality, profiles
core/validation.py             ROI and spacing validation
core/stabilization.py          Consensus helpers for future video mode
core/metrics.py                Evaluation metrics and CSV output
core/storage.py                SQLite history store
templates/index.html           Responsive dashboard
static/css/style.css           Industrial dashboard styling
static/js/app.js               Camera/gallery, compression, ROI and result UI
evaluate.py                    Held-out accuracy evaluation
train.py                       Optional YOLO training entry point
test_counter.py                Automated smoke/regression tests
requirements.txt               CPU/web runtime
requirements-ml.txt            Optional YOLO/Torch training runtime
Procfile                      Gunicorn start command
vercel.json                    Optional Vercel Python runtime configuration
weights/                       Put ring_best.pt here; weights are git-ignored
database/                      Local SQLite database; database files are ignored
results/                       Evaluation output; generated files are ignored
```

## Windows setup and local run

Open PowerShell:

```powershell
cd "C:\Users\GTIS-\Downloads\ring-counter-pro"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. On a phone connected to the same Wi-Fi, find the
PC address with `ipconfig` and open `http://PC_IP:5000`.

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The exact entry point requested by the client is:

```text
C:\Users\GTIS-\Downloads\ring-counter-pro\app.py
```

You can also start it without activating the environment:

```powershell
cd "C:\Users\GTIS-\Downloads\ring-counter-pro"
py -3.11 -m pip install -r requirements.txt
py -3.11 app.py
```

## API

### `GET /status`

Returns server version, device, model-loaded state, and CPU/GPU fallback
status.

### `POST /count`

Multipart fields:

- `photo`: JPG, JPEG, PNG, or WEBP
- `top_frac`: top tap Y divided by image height
- `bottom_frac`: bottom tap Y divided by image height
- `x_frac`: horizontal tap X divided by image width
- `auto_roi=1`: optional full-image fallback; it deliberately reduces
  confidence and is not equivalent to manual ROI selection

Successful responses include `ring_count`, `confidence`, numeric
`confidence_score`, quality metrics, detected/rejected counts, missing-gap
hypotheses, processing time, ROI, and annotated/cropped image data URLs.

### `GET /api/history`

Returns recent results from local SQLite. `GET /api/history.csv` exports them.
On serverless deployments such as Vercel, local SQLite is ephemeral; use a
managed database before relying on history across deployments.

## Tests

Run from the project directory:

```powershell
python -m py_compile app.py config.py train.py evaluate.py
python -m pytest -q
```

The included synthetic test proves that the pipeline, response shape, and
error handling work. It is **not** a production accuracy claim.

## Measuring real accuracy

Create a held-out CSV after collecting manually verified images:

```csv
image,actual,top_frac,bottom_frac,x_frac,auto_roi
test_images/stack_001.jpg,44,0.04,0.76,0.50,0
test_images/stack_002.jpg,47,0.03,0.75,0.50,0
test_images/stack_003.jpg,37,0.02,0.84,0.50,0
```

Keep train, validation, and test images separate. Do not use final test images
to tune thresholds or train the model. Then run:

```powershell
python evaluate.py --manifest dataset/test_manifest.csv --output results/evaluation_results.csv --summary results/evaluation_summary.json
python evaluate.py --manifest dataset/test_manifest.csv --fail-below-target --target 0.90
```

The report contains exact-match accuracy, MAE, mean error, over-count rate,
under-count rate, average confidence, and one row per image. Three images are
not enough for a reliable 90% production model; collect at least 100–150
annotated images, preferably 250–500+ across lighting, angles, counts, ring
finishes, cameras, and backgrounds.

## Optional YOLO model

Put a trained one-class model at:

```text
weights/ring_best.pt
```

Install the optional runtime:

```powershell
python -m pip install -r requirements-ml.txt
```

The server loads the model once at startup. Set `DEVICE=cuda` only on a
machine with a compatible CUDA/Torch installation; `DEVICE=auto` is the
default and falls back to CPU.

For training:

```powershell
python train.py --data dataset/data.yaml --epochs 150 --imgsz 1024 --batch 4 --device auto
```

Use one class named `ring`. The dataset should follow Ultralytics YOLO
detection or segmentation layout and must have independent train/val/test
images.

## Production deployment

### Render, Railway, or a Linux VM (recommended for OpenCV)

Push the project to GitHub, configure Python, and use:

```bash
pip install -r requirements.txt
gunicorn --workers 2 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT app:app
```

The included `Procfile` contains the same start command. For real history,
attach PostgreSQL or another persistent store; the included SQLite store is
appropriate for local/single-instance use.

### Vercel

`vercel.json` is included because the existing repository was deployed there.
After pushing the updated files:

1. In Vercel, open the existing project and redeploy the latest Git commit.
2. Keep the Root Directory at the folder containing `app.py`.
3. Use Python runtime detection from `vercel.json`; do not add a Node build.
4. Verify `/status`, then upload one small image from the public URL.
5. Treat Vercel storage as ephemeral: SQLite history and generated files are
   not durable between serverless instances.

For a client workload with frequent image inference or a YOLO weight file,
prefer Render/Railway/a Linux VM because the process stays warm and the
OpenCV/ML dependency footprint is more predictable. Do not commit secrets,
model credentials, or large weights to GitHub.

## Mobile checks

Open the deployed HTTPS URL on Android Chrome or iPhone Safari:

1. Tap **Take photo** and capture the complete stack.
2. Wait for the compressed preview.
3. Tap the visible stack top, then bottom.
4. Press **Analyze rings**.
5. Review confidence, quality, spacing, and the annotated lines.
6. Download the annotated or crop image.

If the result is low confidence, retake the photo straight-on with less glare,
better focus, and the complete stack inside the frame. The UI's full-image
fallback is for troubleshooting, not a replacement for manual ROI.