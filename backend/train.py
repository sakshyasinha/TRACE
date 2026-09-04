from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ml import SpatialFrequencyDetector, image_to_tensors, iter_image_paths


class ImageDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, root: Path) -> None:
        self.samples = list(iter_image_paths(root))
        if not self.samples:
            raise ValueError(
                f"No images found below {root.resolve()}. "
                "Add images under real/ and generator folders, or run "
                "python make_demo_data.py for a pipeline smoke test."
            )
        labels = {label for _, label in self.samples}
        if labels != {0, 1}:
            raise ValueError(
                f"Expected both real and synthetic images below {root.resolve()}, "
                f"but found labels {sorted(labels)}. See data/README.md."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        path, label = self.samples[index]
        spatial, frequency = image_to_tensors(Image.open(path))
        return spatial, frequency, torch.tensor(float(label))


def train(data_dir: Path, output: Path, epochs: int, batch_size: int) -> None:
    dataset = ImageDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialFrequencyDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for spatial, frequency, labels in loader:
            outputs = model(spatial.to(device), frequency.to(device))
            loss = criterion(outputs["fusion_logit"], labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        print(f"epoch={epoch + 1}/{epochs} loss={running_loss / len(dataset):.4f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "image_size": 224}, output)
    print(f"saved weights to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TRACE spatial + frequency detector")
    parser.add_argument("--data", type=Path, default=Path("data/train"))
    parser.add_argument("--output", type=Path, default=Path("weights/trace.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size)
