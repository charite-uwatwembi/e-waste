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
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):(4173|5173|5500)",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path(
    os.getenv("MODEL_CHECKPOINT", "artifacts/best.pt")
)

MODEL = None
CHECKPOINT = None


def build_model(num_classes):
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        num_classes,
    )
    return model


def load_model():
    global MODEL, CHECKPOINT

    if MODEL is not None:
        return MODEL, CHECKPOINT

    if not CHECKPOINT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Checkpoint not found: {CHECKPOINT_PATH}",
        )

    CHECKPOINT = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    MODEL = build_model(
        num_classes=len(CHECKPOINT["class_names"])
    )

    MODEL.load_state_dict(CHECKPOINT["model_state"])
    MODEL.to(DEVICE)
    MODEL.eval()

    return MODEL, CHECKPOINT


def predict_image(image):
    model, checkpoint = load_model()

    transform = transforms.Compose([
        transforms.Resize(
            (
                checkpoint["image_size"],
                checkpoint["image_size"],
            )
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    image_tensor = transform(
        image.convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(
            model(image_tensor),
            dim=1,
        )[0]

    values, indices = torch.topk(
        probabilities,
        k=min(3, len(checkpoint["class_names"])),
    )

    return [
        {
            "label": checkpoint["class_names"][index.item()],
            "confidence": round(value.item(), 4),
        }
        for value, index in zip(values, indices)
    ]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": CHECKPOINT_PATH.exists(),
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    try:
        image = Image.open(
            BytesIO(await file.read())
        ).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from error

    predictions = predict_image(image)

    return {
        "predictions": predictions,
    }