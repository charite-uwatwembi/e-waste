import os
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


app = FastAPI(
    title="Kigali E-waste Classifier API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=(
        r"^http://(localhost|127\.0\.0\.1):"
        r"(4173|5173|5174|5500)$"
    ),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# app.py is inside backend/, so parent.parent is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "best.pt"
)

CHECKPOINT_PATH = Path(
    os.getenv(
        "MODEL_CHECKPOINT",
        str(DEFAULT_CHECKPOINT),
    )
)

MODEL = None
CHECKPOINT = None


def build_model(num_classes: int):
    """Create the same MobileNetV3-Small architecture used in training."""

    model = mobilenet_v3_small(weights=None)

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        num_classes,
    )

    return model


def load_model():
    """Load the trained checkpoint the first time it is needed."""

    global MODEL, CHECKPOINT

    if MODEL is not None:
        return MODEL, CHECKPOINT

    if not CHECKPOINT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model checkpoint was not found at: "
                f"{CHECKPOINT_PATH}"
            ),
        )

    try:
        CHECKPOINT = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
        )

        MODEL = build_model(
            num_classes=len(CHECKPOINT["class_names"])
        )

        MODEL.load_state_dict(
            CHECKPOINT["model_state"]
        )

        MODEL.to(DEVICE)
        MODEL.eval()

    except Exception as error:
        MODEL = None
        CHECKPOINT = None

        raise HTTPException(
            status_code=500,
            detail=f"Could not load the model: {error}",
        ) from error

    return MODEL, CHECKPOINT


def predict_image(image: Image.Image):
    """Preprocess an image and return the three best predictions."""

    model, checkpoint = load_model()

    image_size = checkpoint.get("image_size", 224)
    class_names = checkpoint["class_names"]

    transform = transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    image_tensor = (
        transform(image.convert("RGB"))
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.inference_mode():
        logits = model(image_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    values, indices = torch.topk(
        probabilities,
        k=min(3, len(class_names)),
    )

    predictions = []

    for value, index in zip(values, indices):
        predictions.append({
            "label": class_names[index.item()],
            "confidence": round(value.item(), 4),
        })

    return predictions


@app.get("/")
def root():
    return {
        "message": "Kigali E-waste Classifier API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": CHECKPOINT_PATH.exists(),
        "device": str(DEVICE),
        "checkpoint": str(CHECKPOINT_PATH),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    try:
        file_bytes = await file.read()

        image = Image.open(
            BytesIO(file_bytes)
        ).convert("RGB")

    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a readable image.",
        ) from error

    predictions = predict_image(image)

    return {
        "filename": file.filename,
        "predictions": predictions,
    }