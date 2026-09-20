from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from ml import Detector, image_to_tensors, iter_image_paths


def label_for_path(path: Path) -> tuple[int, str]:
    parts = {part.lower() for part in path.parts}
    if "real" in parts:
        return 0, "real"
    if "fake" in parts or "synthetic" in parts or "generator_a" in parts:
        return 1, "synthetic"
    raise ValueError(f"Cannot infer label from path: {path}")


def inspect_preprocessing(path: Path, image_size: int) -> None:
    with Image.open(path) as image:
        exif = image.getexif()
        print("PREPROCESSING")
        print(json.dumps({
            "path": str(path),
            "original_size": image.size,
            "original_mode": image.mode,
            "exif_orientation": exif.get(274),
            "alpha_present": "A" in image.getbands(),
            "crop": "none",
            "resize": [image_size, image_size],
            "conversion": "Pillow convert RGB; no EXIF transpose",
            "normalization": "(pixel / 255 - 0.5) / 0.5",
        }, indent=2))
        spatial, frequency = image_to_tensors(image, image_size)
    print(json.dumps({
        "spatial_shape": list(spatial.shape),
        "spatial_dtype": str(spatial.dtype),
        "spatial_min": float(spatial.min()),
        "spatial_max": float(spatial.max()),
        "spatial_mean": float(spatial.mean()),
        "spatial_std": float(spatial.std()),
        "frequency_shape": list(frequency.shape),
        "frequency_dtype": str(frequency.dtype),
        "frequency_min": float(frequency.min()),
        "frequency_max": float(frequency.max()),
        "frequency_mean": float(frequency.mean()),
        "frequency_std": float(frequency.std()),
    }, indent=2))


def inspect_model(detector: Detector, path: Path) -> dict[str, float | str]:
    with Image.open(path) as image:
        spatial, frequency = image_to_tensors(image, detector.image_size)
    with torch.inference_mode():
        outputs = detector.model(spatial.unsqueeze(0).to(detector.device), frequency.unsqueeze(0).to(detector.device))
    spatial_logit = float(outputs["spatial_logit"][0])
    frequency_logit = float(outputs["frequency_logit"][0])
    fusion_logit = float(outputs["fusion_logit"][0])
    spatial_probability = float(torch.sigmoid(outputs["spatial_logit"])[0])
    frequency_probability = float(torch.sigmoid(outputs["frequency_logit"])[0])
    fusion_probability = float(torch.sigmoid(outputs["fusion_logit"])[0])
    fixed_probability_average = (spatial_probability + frequency_probability) / 2
    fixed_logit_average_probability = float(torch.sigmoid(torch.tensor((spatial_logit + frequency_logit) / 2)))
    return {
        "spatial_logit": spatial_logit,
        "spatial_probability": spatial_probability,
        "frequency_logit": frequency_logit,
        "frequency_probability": frequency_probability,
        "learned_fusion_logit": fusion_logit,
        "learned_fusion_probability": fusion_probability,
        "fixed_50_50_probability_average": fixed_probability_average,
        "fixed_50_50_logit_average_probability": fixed_logit_average_probability,
        "spatial_direction": "synthetic" if spatial_probability >= 0.5 else "authentic",
        "frequency_direction": "synthetic" if frequency_probability >= 0.5 else "authentic",
        "learned_fusion_direction": "synthetic" if fusion_probability >= 0.5 else "authentic",
    }


def sample_paths(root: Path, folder: str, count: int) -> list[Path]:
    return sorted((root / folder).glob("*.jpg"))[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose TRACE without changing model behavior")
    parser.add_argument("--weights", type=Path, default=Path("weights/trace-baseline-64.pt"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--personal", type=Path, default=None)
    args = parser.parse_args()

    detector = Detector(args.weights)
    print("MODEL")
    print(json.dumps({
        "weights": str(args.weights),
        "ready": detector.ready,
        "image_size": detector.image_size,
        "sample_count": detector.sample_count,
        "quality": detector.quality_label,
        "sigmoid": "torch.sigmoid(raw branch/fusion logit)",
        "threshold": 0.5,
        "fusion_formula": "sigmoid(MLP(concat(spatial_embedding, frequency_embedding)))",
        "fixed_comparison_formula": "average(sigmoid(spatial_logit), sigmoid(frequency_logit)) and sigmoid(average(logits))",
    }, indent=2))

    paths = [
        ("train_real", sample_paths(args.data / "train", "real", 10)),
        ("train_fake", sample_paths(args.data / "train", "FAKE", 10)),
        ("test_real", sample_paths(args.data / "test", "REAL", 10)),
        ("test_fake", sample_paths(args.data / "test", "FAKE", 10)),
    ]
    if args.personal:
        paths.append(("personal", [args.personal]))

    print("SANITY_SET")
    print("name,ground_truth,spatial_logit,spatial_probability,frequency_logit,frequency_probability,learned_fusion_logit,learned_fusion_probability,fixed_probability_average,fixed_logit_average_probability")
    for name, group in paths:
        for path in group:
            ground_truth = label_for_path(path)[1] if name != "personal" else "real (declared)"
            values = inspect_model(detector, path)
            print(",".join([
                name,
                ground_truth,
                *(f"{values[key]:.6f}" for key in [
                    "spatial_logit",
                    "spatial_probability",
                    "frequency_logit",
                    "frequency_probability",
                    "learned_fusion_logit",
                    "learned_fusion_probability",
                    "fixed_50_50_probability_average",
                    "fixed_50_50_logit_average_probability",
                ]),
            ]))

    if args.personal:
        inspect_preprocessing(args.personal, detector.image_size)


if __name__ == "__main__":
    main()
