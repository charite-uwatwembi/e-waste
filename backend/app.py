import os
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from ml.predict import load_checkpoint, predict_pil


app = FastAPI(
    title="Kigali E-waste Classifier API",
    version="0.2.0",
)


# Allow the React frontend to communicate with FastAPI.
LOCAL_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")

if FRONTEND_ORIGIN:
    LOCAL_ORIGINS.append(
        FRONTEND_ORIGIN.rstrip("/")
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCAL_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = Path(
    os.getenv(
        "MODEL_CHECKPOINT",
        str(PROJECT_ROOT / "artifacts" / "best.pt"),
    )
)

# Predictions below this confidence are treated as uncertain.
CONFIDENCE_THRESHOLD = float(
    os.getenv("CONFIDENCE_THRESHOLD", "0.60")
)

MODEL = None
CHECKPOINT = None


# ---------------------------------------------------------
# Kinyarwanda names and instructions
# ---------------------------------------------------------

CLASS_GUIDANCE = {
    "Battery": {
        "label_rw": "Batiri",
        "message": "Iki gikoresho ni batiri.",
        "next_step": (
            "Yishyire ukwazo kandi uyijyane ahakusanyirizwa "
            "ibisigazwa by'ibikoresho by'ikoranabuhanga. "
            "Ntuyivange n'imyanda isanzwe."
        ),
        "warning": (
            "Ntuyimene, ntuyitwike kandi ntuyifungure."
        ),
    },

    "Keyboard": {
        "label_rw": "Mwandikisho ya mudasobwa",
        "message": (
            "Iki gikoresho ni mwandikisho ya mudasobwa."
        ),
        "next_step": (
            "Kijyane ahakusanyirizwa ibisigazwa "
            "by'ibikoresho by'ikoranabuhanga. "
            "Ntukivange n'imyanda isanzwe."
        ),
        "warning": (
            "Niba cyarangiritse, ntukagifungure "
            "cyangwa ngo ugitwike."
        ),
    },

    "Microwave": {
        "label_rw": "Ifuru ya mikorowave",
        "message": (
            "Iki gikoresho ni ifuru ya mikorowave."
        ),
        "next_step": (
            "Kibike uko cyakabaye kandi ugishyikirize "
            "ahakusanyirizwa ibisigazwa by'ibikoresho "
            "by'ikoranabuhanga."
        ),
        "warning": (
            "Ntukagifungure cyangwa ngo ugisenye."
        ),
    },

    "Mobile": {
        "label_rw": "Telefoni igendanwa",
        "message": (
            "Iki gikoresho ni telefoni igendanwa."
        ),
        "next_step": (
            "Kijyane ahakusanyirizwa ibisigazwa "
            "by'ibikoresho by'ikoranabuhanga. "
            "Ntukijugunye mu myanda isanzwe."
        ),
        "warning": (
            "Ntukagifungure cyangwa ngo ugitwike, "
            "kuko gishobora kuba kirimo batiri."
        ),
    },

    "Mouse": {
        "label_rw": "Imbeba ya mudasobwa",
        "message": (
            "Iki gikoresho ni imbeba ya mudasobwa."
        ),
        "next_step": (
            "Kijyane ahakusanyirizwa ibisigazwa "
            "by'ibikoresho by'ikoranabuhanga. "
            "Ntukivange n'imyanda isanzwe."
        ),
        "warning": (
            "Niba kirimo batiri, ntukimene cyangwa "
            "ngo ugishyire mu muriro."
        ),
    },

    "PCB": {
        "label_rw": (
            "Ikibaho cy'umuzunguruko wa elegitoroniki"
        ),
        "message": (
            "Iki ni ikibaho cy'umuzunguruko "
            "wa elegitoroniki."
        ),
        "next_step": (
            "Gishyire hamwe n'ibisigazwa by'ibikoresho "
            "by'ikoranabuhanga kandi ugishyikirize "
            "ahabugenewe."
        ),
        "warning": (
            "Ntukagitwike cyangwa ngo ukimene."
        ),
    },

    "Player": {
        "label_rw": (
            "Igikoresho gikina amajwi cyangwa amashusho"
        ),
        "message": (
            "Iki ni igikoresho gikina amajwi "
            "cyangwa amashusho."
        ),
        "next_step": (
            "Kijyane ahakusanyirizwa ibisigazwa "
            "by'ibikoresho by'ikoranabuhanga. "
            "Ntukivange n'imyanda isanzwe."
        ),
        "warning": (
            "Niba kirimo batiri, ntukagifungure "
            "cyangwa ngo ugitwike."
        ),
    },

    "Printer": {
        "label_rw": "Mucapyi",
        "message": (
            "Iki gikoresho ni mucapyi."
        ),
        "next_step": (
            "Kijyane ahakusanyirizwa ibisigazwa "
            "by'ibikoresho by'ikoranabuhanga. "
            "Niba kirimo wino, gikomeze guhagarara "
            "neza kugira ngo itameneka."
        ),
        "warning": (
            "Ntukagitwike cyangwa ngo ugisenye."
        ),
    },

    "Television": {
        "label_rw": "Televiziyo",
        "message": (
            "Iki gikoresho ni televiziyo."
        ),
        "next_step": (
            "Kibike uko cyakabaye kandi ugishyikirize "
            "ahakusanyirizwa ibisigazwa by'ibikoresho "
            "by'ikoranabuhanga."
        ),
        "warning": (
            "Ntukagisenye kandi wirinde ikirahure "
            "cyacyo niba cyaramenetse."
        ),
    },

    "Washing Machine": {
        "label_rw": "Imashini imesa imyenda",
        "message": (
            "Iki gikoresho ni imashini imesa imyenda."
        ),
        "next_step": (
            "Kibike ukwacyo kandi ushake ubufasha "
            "bwo kukijyana ahakusanyirizwa ibisigazwa "
            "by'ibikoresho by'ikoranabuhanga."
        ),
        "warning": (
            "Ntukagisenye; saba ubufasha mu kukimura "
            "niba kiremereye."
        ),
    },
}


UNKNOWN_GUIDANCE = {
    "label_rw": "Ntibimenyekanye neza",
    "message": (
        "Iki gikoresho nticyamenyekanye neza."
    ),
    "next_step": (
        "Ntukijugunye mu myanda isanzwe. "
        "Baza umukozi ubifitiye ubumenyi kugira ngo "
        "agenzure iki gikoresho."
    ),
    "warning": (
        "Ntukimene, ntukagitwike kandi ntugikoreho "
        "niba gishyushye cyangwa gitemba."
    ),
}


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

def get_model():
    global MODEL, CHECKPOINT

    if MODEL is not None:
        return MODEL, CHECKPOINT

    if not CHECKPOINT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model checkpoint was not found at "
                f"{CHECKPOINT_PATH}. Train the model first."
            ),
        )

    try:
        MODEL, CHECKPOINT = load_checkpoint(
            CHECKPOINT_PATH,
            DEVICE,
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load the model: {error}",
        ) from error

    return MODEL, CHECKPOINT


# ---------------------------------------------------------
# API routes
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Kigali E-waste Classifier API",
        "docs": "/docs",
        "health": "/health",
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
async def predict(
    file: UploadFile = File(...),
):
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    try:
        file_content = await file.read()

        image = Image.open(
            BytesIO(file_content)
        ).convert("RGB")

    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file is not a valid image."
            ),
        ) from error

    model, checkpoint = get_model()

    predictions = predict_pil(
        image=image,
        model=model,
        class_names=checkpoint["class_names"],
        image_size=checkpoint["image_size"],
    )

    if not predictions:
        raise HTTPException(
            status_code=500,
            detail="The model did not return a prediction.",
        )

    top_prediction = predictions[0]

    label_en = top_prediction["label"]
    confidence = top_prediction["confidence"]


    # Handle uncertain predictions.
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "prediction": {
                "label_en": "Unknown",
                "label_rw": UNKNOWN_GUIDANCE["label_rw"],
                "confidence": confidence,
            },
            "guidance_rw": {
                "message": UNKNOWN_GUIDANCE["message"],
                "next_step": UNKNOWN_GUIDANCE["next_step"],
                "warning": UNKNOWN_GUIDANCE["warning"],
            },
        }


    # Find the Kinyarwanda message for the prediction.
    guidance = CLASS_GUIDANCE.get(
        label_en,
        UNKNOWN_GUIDANCE,
    )

    return {
        "prediction": {
            "label_en": label_en,
            "label_rw": guidance["label_rw"],
            "confidence": confidence,
        },
        "guidance_rw": {
            "message": guidance["message"],
            "next_step": guidance["next_step"],
            "warning": guidance["warning"],
        },
    }