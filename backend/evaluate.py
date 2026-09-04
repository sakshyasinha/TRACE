from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from ml import Detector, image_to_tensors, iter_image_paths


def accuracy(detector: Detector, root: Path) -> float:
    samples = list(iter_image_paths(root))
    if not samples:
        raise ValueError(f"No images found below {root}")
    correct = 0
    for path, label in samples:
        spatial, frequency = image_to_tensors(Image.open(path))
        with torch.inference_mode():
            outputs = detector.model(spatial.unsqueeze(0).to(detector.device), frequency.unsqueeze(0).to(detector.device))
            prediction = int(torch.sigmoid(outputs["fusion_logit"])[0] >= 0.5)
        correct += prediction == label
    return correct / len(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TRACE on a named split")
    parser.add_argument("split", type=Path, help="Directory containing real and generator folders")
    parser.add_argument("--weights", type=Path, default=Path("weights/trace.pt"))
    args = parser.parse_args()
    detector = Detector(args.weights)
    if not detector.ready:
        raise RuntimeError("No trained checkpoint found. Run backend/train.py first.")
    print(f"split={args.split} accuracy={accuracy(detector, args.split):.4f}")