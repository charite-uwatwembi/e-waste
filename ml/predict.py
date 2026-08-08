from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL.Image import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(num_classes: int) -> torch.nn.Module:
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("model_name", "mobilenet_v3_small")

    if model_name != "mobilenet_v3_small":
        raise ValueError(f"Unsupported model_name: {model_name}")

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, checkpoint


def preprocess_image(image: Image, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transform(image).unsqueeze(0)


def predict_pil(
    image: Image,
    model: torch.nn.Module,
    class_names: List[str],
    image_size: int,
) -> List[Dict[str, Any]]:
    tensor = preprocess_image(image, image_size)
    tensor = tensor.to(next(model.parameters()).device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    return sorted(
        [
            {
                "label": label,
                "confidence": float(confidence),
            }
            for label, confidence in zip(class_names, probabilities)
        ],
        key=lambda item: item["confidence"],
        reverse=True,
    )
