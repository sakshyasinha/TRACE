from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch import Tensor, nn

IMAGE_SIZE = 224


def image_to_tensors(image: Image.Image, image_size: int = IMAGE_SIZE) -> tuple[Tensor, Tensor]:
    image = image.convert("RGB").resize((image_size, image_size))
    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    spatial = pixels.reshape(image_size, image_size, 3).permute(2, 0, 1).float() / 255.0
    spatial = (spatial - 0.5) / 0.5
    grayscale = spatial.mean(dim=0, keepdim=True)
    spectrum = torch.fft.fftshift(torch.fft.fft2(grayscale))
    frequency = torch.log1p(torch.abs(spectrum))
    frequency = (frequency - frequency.mean()) / (frequency.std() + 1e-6)
    return spatial, frequency


class ConvBranch(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Linear(64, 32)
        self.head = nn.Linear(32, 1)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        embedding = self.embedding(self.features(inputs).flatten(1))
        return embedding, self.head(embedding)


class SpatialFrequencyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial = ConvBranch(3)
        self.frequency = ConvBranch(1)
        self.fusion = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, spatial: Tensor, frequency: Tensor) -> dict[str, Tensor]:
        spatial_embedding, spatial_logit = self.spatial(spatial)
        frequency_embedding, frequency_logit = self.frequency(frequency)
        fusion_logit = self.fusion(torch.cat((spatial_embedding, frequency_embedding), dim=1))
        return {
            "spatial_logit": spatial_logit.squeeze(1),
            "frequency_logit": frequency_logit.squeeze(1),
            "fusion_logit": fusion_logit.squeeze(1),
        }


@dataclass(frozen=True)
class Prediction:
    label: str
    synthetic_likelihood: float
    spatial_signal: float
    frequency_signal: float
    compression_response: float
    status: str
    note: str


class Detector:
    def __init__(self, weights_path: Path) -> None:
        self.weights_path = weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: SpatialFrequencyDetector | None = None
        if weights_path.exists():
            self.model = SpatialFrequencyDetector().to(self.device)
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()

    @property
    def ready(self) -> bool:
        return self.model is not None

    @torch.inference_mode()
    def predict(self, image_bytes: bytes, filename: str | None) -> Prediction:
        if self.model is None:
            raise RuntimeError("trained weights are not available; run backend/train.py first")
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        spatial, frequency = image_to_tensors(image)
        outputs = self.model(spatial.unsqueeze(0).to(self.device), frequency.unsqueeze(0).to(self.device))
        likelihood = float(torch.sigmoid(outputs["fusion_logit"])[0])
        spatial_signal = float(torch.sigmoid(outputs["spatial_logit"])[0])
        frequency_signal = float(torch.sigmoid(outputs["frequency_logit"])[0])
        compressed = BytesIO()
        image.save(compressed, format="JPEG", quality=70)
        compressed_spatial, compressed_frequency = image_to_tensors(Image.open(BytesIO(compressed.getvalue())))
        compressed_outputs = self.model(compressed_spatial.unsqueeze(0).to(self.device), compressed_frequency.unsqueeze(0).to(self.device))
        compressed_likelihood = float(torch.sigmoid(compressed_outputs["fusion_logit"])[0])
        label = "likely synthetic" if likelihood >= 0.5 else "likely authentic"
        return Prediction(label, likelihood, spatial_signal, frequency_signal, abs(likelihood - compressed_likelihood), "inference", f"Analyzed {filename or 'uploaded image'} with the fused spatial + frequency model.")


def iter_image_paths(root: Path) -> Iterable[tuple[Path, int]]:
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            label = 0 if "real" in {part.lower() for part in path.parts} else 1
            yield path, label
