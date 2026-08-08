# Kigali E-waste Classifier

An AI-assisted e-waste classification web app for informal waste pickers in Kigali.

The first milestone is a lightweight image classifier that predicts the type of e-waste in a photo. The project is deliberately split into stages:

1. Train a baseline classifier on the Kaggle image dataset.
2. Validate it on photos collected in Kigali-like conditions (outdoor light, mixed backgrounds, damaged devices).
3. Add a FastAPI prediction service and a mobile-friendly web interface.
4. Later, train an object detector on the Roboflow dataset for photos containing multiple devices.

## Recommended model scope

The first release should classify a small set of actionable categories rather than claim to identify every electronic device or determine whether an item is safe. Predictions are decision support only. A human should handle batteries, CRTs, suspected refrigerants, broken screens, and unknown items using approved safety procedures.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Download and extract the Kaggle dataset into `data/raw/e-waste-image-dataset/` so that each class is a directory containing images. The expected shape is:

```text
data/raw/e-waste-image-dataset/
├── Battery/
├── Charger/
├── Keyboard/
└── ...
```

Do not commit the images to Git. Keep the original dataset license and citation with your project records.

## Inspect and train

```bash
python -m ml.inspect_dataset --data-dir data/raw/e-waste-image-dataset
python -m ml.train --data-dir data/raw/e-waste-image-dataset --epochs 15
```

The training command writes the best checkpoint and evaluation outputs to `artifacts/`:

```text
artifacts/
├── best.pt
├── class_report.json
├── confusion_matrix.png
└── history.json
```

Test a single image:

```bash
python -m ml.predict --checkpoint artifacts/best.pt --image path/to/photo.jpg
```

## Run the web app

Start the API in one terminal:

```bash
MODEL_CHECKPOINT=artifacts/best.pt uvicorn backend.app:app --reload --port 8000
```

Serve the frontend in a second terminal:

```bash
python -m http.server 5500 --directory frontend
```

Open `http://localhost:5500` and upload an image. The frontend sends it to `http://localhost:8000/predict`.

## Dataset choice

The Kaggle dataset is the right starting point for this repository because it is a classification dataset with roughly 2,400 labeled images across about nine or ten device categories. It enables a fast end-to-end baseline.

The Roboflow dataset is better for a later detection milestone: its public page reports about 19,613 annotated images and a broad device vocabulary, with bounding-box and polygon annotations. It also reports 43 classes on the current model view while the project description says 77 classes, so the exact exported version and class list must be recorded before training. It is licensed CC BY 4.0 on the project page and aggregates images from many source datasets; preserve attribution and audit the source licenses before redistribution.

## Project structure

```text
.
├── backend/
│   └── app.py                 # FastAPI upload and prediction endpoint
├── data/
│   └── raw/                   # Local, untracked datasets
├── docs/
│   └── PROJECT_PLAN.md        # Scope, labels, validation, and roadmap
├── frontend/
│   ├── index.html             # Upload page
│   ├── app.js                 # API call and result rendering
│   └── styles.css             # Mobile-friendly styling
├── ml/
│   ├── data.py                # ImageFolder loading and stratified splits
│   ├── inspect_dataset.py     # Class counts and data sanity checks
│   ├── model.py               # MobileNetV3-small transfer-learning model
│   ├── predict.py             # Checkpoint loading and inference helpers
│   └── train.py               # Training, evaluation, and artifact export
├── artifacts/                 # Generated checkpoints and metrics
├── requirements.txt
└── README.md
```
