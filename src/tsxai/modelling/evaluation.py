from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    compute_confusion_matrix: bool = False,
) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    """Evaluates a PyTorch model using streaming metrics computation.

    Uses torchmetrics for efficient, GPU-accelerated metric computation.
    Metrics are computed on-the-fly without storing all predictions in memory.
    Calculates accuracy and F1-score by default, with optional confusion matrix.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        num_classes: Number of classes in the classification task
        compute_confusion_matrix: If True, computes and returns confusion matrix.
            Defaults to False to save memory.

    Returns:
        Tuple containing:
        - Dictionary of metric names and values
        - Confusion matrix as numpy array if compute_confusion_matrix=True,
          otherwise None
    """
    model.to(device)
    model.eval()

    # Initialize metrics
    metrics = {
        "accuracy": torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        ).to(device),
        "f1": torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        ).to(device),
    }

    # Optionally add confusion matrix
    if compute_confusion_matrix:
        metrics["confusion_matrix"] = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        ).to(device)

    # Evaluation loop
    with torch.no_grad():
        for batch, target in dataloader:
            batch, target = batch.to(device), target.to(device)
            output = model(batch)
            # Update metrics
            for metric in metrics.values():
                metric.update(output, target)

    # Compute final metrics
    results = {}
    conf_matrix = None
    for name, metric in metrics.items():
        if name == "confusion_matrix":
            conf_matrix = metric.compute().cpu().numpy()
        else:
            results[name] = metric.compute().item()

    return (results, conf_matrix) if compute_confusion_matrix else results
