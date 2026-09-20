from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml import Detector, image_to_tensors, iter_image_paths

METHODS = {
    "spatial_only": "Spatial-only",
    "frequency_only": "Frequency-only",
    "fixed_fusion": "Fixed 50/50",
    "learned_fusion": "Learned fusion",
}


def source_name(path: Path, root: Path) -> str:
    relative_parts = path.relative_to(root).parts
    return relative_parts[0] if len(relative_parts) > 1 else "unspecified"


def stratified_split(samples: list[tuple[Path, int]], validation_fraction: float, seed: int) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    randomizer = random.Random(seed)
    by_label: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for sample in samples:
        by_label[sample[1]].append(sample)
    validation: list[tuple[Path, int]] = []
    training: list[tuple[Path, int]] = []
    for label, label_samples in by_label.items():
        randomizer.shuffle(label_samples)
        validation_count = max(1, int(len(label_samples) * validation_fraction))
        validation.extend(label_samples[:validation_count])
        training.extend(label_samples[validation_count:])
    return sorted(training), sorted(validation)


@torch.inference_mode()
def collect_outputs(detector: Detector, samples: Iterable[tuple[Path, int]], root: Path, split_name: str, batch_size: int = 128) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    sample_list = list(samples)
    for start in range(0, len(sample_list), batch_size):
        batch = sample_list[start:start + batch_size]
        spatial_inputs: list[torch.Tensor] = []
        frequency_inputs: list[torch.Tensor] = []
        for path, _ in batch:
            with Image.open(path) as image:
                spatial, frequency = image_to_tensors(image, detector.image_size)
            spatial_inputs.append(spatial)
            frequency_inputs.append(frequency)
        outputs = detector.model(
            torch.stack(spatial_inputs).to(detector.device),
            torch.stack(frequency_inputs).to(detector.device),
        )
        spatial_logits = outputs["spatial_logit"]
        frequency_logits = outputs["frequency_logit"]
        fusion_logits = outputs["fusion_logit"]
        spatial_scores = torch.sigmoid(spatial_logits)
        frequency_scores = torch.sigmoid(frequency_logits)
        fusion_scores = torch.sigmoid(fusion_logits)
        for index, (path, label) in enumerate(batch):
            spatial_logit = float(spatial_logits[index])
            frequency_logit = float(frequency_logits[index])
            fusion_logit = float(fusion_logits[index])
            spatial_score = float(spatial_scores[index])
            frequency_score = float(frequency_scores[index])
            fusion_score = float(fusion_scores[index])
            records.append({
                "path": str(path),
                "filename": path.name,
                "split": split_name,
                "source": source_name(path, root),
                "label": label,
                "label_name": "synthetic" if label else "real",
                "spatial_logit": spatial_logit,
                "spatial_score": spatial_score,
                "frequency_logit": frequency_logit,
                "frequency_score": frequency_score,
                "fusion_logit": fusion_logit,
                "fusion_score": fusion_score,
                "fixed_probability_average": (spatial_score + frequency_score) / 2,
                "fixed_logit_average_probability": float(torch.sigmoid(torch.tensor((spatial_logit + frequency_logit) / 2))),
            })
    return records


def method_scores(record: dict[str, object], method: str) -> float:
    return float(record[{
        "spatial_only": "spatial_score",
        "frequency_only": "frequency_score",
        "fixed_fusion": "fixed_probability_average",
        "learned_fusion": "fusion_score",
    }[method]])


def select_threshold(records: list[dict[str, object]], method: str) -> float:
    labels = [int(record["label"]) for record in records]
    scores = [method_scores(record, method) for record in records]
    candidates = sorted({0.0, 0.5, 1.0, *scores})
    ranked = [(balanced_accuracy_score(labels, [int(score >= threshold) for score in scores]), -abs(threshold - 0.5), threshold) for threshold in candidates]
    return max(ranked)[2]


def metrics(records: list[dict[str, object]], method: str, threshold: float) -> dict[str, object]:
    labels = [int(record["label"]) for record in records]
    scores = [method_scores(record, method) for record in records]
    predictions = [int(score >= threshold) for score in scores]
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
    return {
        "method": method,
        "threshold": threshold,
        "roc_auc": roc_auc_score(labels, scores) if len(set(labels)) == 2 else None,
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": matrix,
    }


def distributions(records: list[dict[str, object]], method: str) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for label, label_name in [(0, "real"), (1, "synthetic")]:
        values = sorted(method_scores(record, method) for record in records if int(record["label"]) == label)
        if not values:
            continue
        midpoint = len(values) // 2
        median = values[midpoint] if len(values) % 2 else (values[midpoint - 1] + values[midpoint]) / 2
        mean = sum(values) / len(values)
        output[label_name] = {
            "mean": mean,
            "median": median,
            "std": (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5,
            "min": values[0],
            "max": values[-1],
        }
    return output


def evaluate_group(records: list[dict[str, object]], methods: list[str], thresholds: dict[str, float]) -> dict[str, object]:
    return {method: {"metrics": metrics(records, method, thresholds[method]), "distributions": distributions(records, method)} for method in methods}


def write_records(records: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".json").open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
    if records:
        with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def print_summary(test_results: dict[str, object]) -> None:
    print("\nMethod             ROC-AUC   Bal.Acc   F1")
    print("------------------------------------------------")
    for method, display_name in METHODS.items():
        result = test_results[method]["metrics"]
        print(f"{display_name:<18} {result['roc_auc'] if result['roc_auc'] is not None else float('nan'):.4f}    {result['balanced_accuracy']:.4f}    {result['f1']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Research evaluation for TRACE's existing detector")
    parser.add_argument("split", type=Path, help="Untouched test split, such as data/test")
    parser.add_argument("--train-root", type=Path, default=Path("data/train"), help="Training pool used only to create validation data")
    parser.add_argument("--weights", type=Path, default=Path("weights/trace.pt"))
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    detector = Detector(args.weights)
    if not detector.ready:
        raise RuntimeError("No trained checkpoint found. Run backend/train.py first.")
    train_samples = list(iter_image_paths(args.train_root))
    test_samples = list(iter_image_paths(args.split))
    if not train_samples or not test_samples:
        raise ValueError("Both --train-root and split must contain images.")
    _, validation_samples = stratified_split(train_samples, args.validation_fraction, args.seed)
    methods = list(METHODS)
    validation_records = collect_outputs(detector, validation_samples, args.train_root, "validation", args.batch_size)
    test_records = collect_outputs(detector, test_samples, args.split, "test", args.batch_size)
    thresholds = {method: select_threshold(validation_records, method) for method in methods}
    validation_results = evaluate_group(validation_records, methods, thresholds)
    test_results = evaluate_group(test_records, methods, thresholds)
    generator_results = {source: evaluate_group([record for record in test_records if record["source"] == source], methods, thresholds) for source in sorted({str(record["source"]) for record in test_records})}
    write_records(validation_records + test_records, args.output.with_name(args.output.name + "_records"))
    report = {
        "weights": str(args.weights),
        "train_root": str(args.train_root),
        "test_root": str(args.split),
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "image_size": detector.image_size,
        "label_mapping": {"real": 0, "synthetic": 1},
        "methods": METHODS,
        "thresholds_selected_on_validation_only": thresholds,
        "validation": validation_results,
        "test": test_results,
        "per_source_test": generator_results,
        "source_split_limitation": "Folder names are reported as sources only; no generator identity is inferred. The current data/test REAL/FAKE layout cannot establish an unseen-generator result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(test_results)
    print("\nInterpretation")
    print("Frequency contains useful class information only if its ROC-AUC and balanced accuracy exceed the trivial/spatial baselines on the untouched test set.")
    print("Fixed fusion improves over spatial-only only when its held-out metrics are higher, not because its score is numerically larger.")
    print("Learned fusion improves over fixed fusion only when the same validation-selected threshold protocol shows a held-out gain.")
    print("Current folder layout reports sources, but does not prove Generator C was unseen.")


if __name__ == "__main__":
    main()