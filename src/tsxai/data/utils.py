import random

import numpy as np
import torch

from tsxai.utils.logging import setup_logger

# Create logger for this module
logger = setup_logger(__name__)


def sample_equal_per_class(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    random_seed: int = 42,
) -> np.ndarray:
    """Sample a fixed number of test IDs from each stratum (class).

    If any class has fewer samples than requested, reduces sampling
    to the size of the smallest class.

    Args:
        X: Input data array.
        y: Target labels array (1D).
        n_samples: Number of samples to select from each class. If any class has
            fewer samples, will reduce to the size of the smallest class.
        random_seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        Array of sampled indices.

    Raises:
        ValueError: If n_samples < 1.
            If X and y have different lengths.
            If no samples are available.
    """
    # Input validation
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    if len(X) != len(y):
        raise ValueError(f"X and y must have same length, got {len(X)} and {len(y)}")

    if len(X) == 0:
        raise ValueError("No samples available for sampling")

    # Set random seed
    np.random.seed(random_seed)

    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(unique_classes)

    logger.info(f"Requested {n_samples} samples from each of {n_classes} classes")

    # Check if we have enough samples in each class
    min_class_count = min(class_counts)
    actual_samples_per_class = min(n_samples, min_class_count)

    if actual_samples_per_class < n_samples:
        logger.warning(
            f"Requested {n_samples} samples per class but smallest class "
            f"only has {min_class_count} samples. Reducing to {actual_samples_per_class} "
            "samples per class."
        )

    # Log available samples per class
    class_dist = (
        "["
        + ", ".join([f"class {c}: {n}" for c, n in zip(unique_classes, class_counts)])
        + "]"
    )
    logger.info(f"Available samples per class: {class_dist}")

    # Sample from each class
    sampled_ids = []
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        class_samples = np.random.choice(
            class_indices, size=actual_samples_per_class, replace=False
        )
        sampled_ids.extend(class_samples)

    sampled_ids = np.array(sampled_ids)

    # Log final distribution
    final_classes, final_counts = np.unique(y[sampled_ids], return_counts=True)
    sample_dist = (
        "["
        + ", ".join([f"class {c}: {n}" for c, n in zip(final_classes, final_counts)])
        + "]"
    )
    logger.info(f"Final sampling distribution: {sample_dist}")

    return sampled_ids


def set_random_seed(seed: int) -> None:
    """Set the random seed for reproducibility across various libraries.

    This function sets the random seed for Python's built-in random module,
    NumPy, and PyTorch (both CPU and CUDA). It also ensures deterministic
    behavior for PyTorch's CUDNN backend.

    Args:
        seed: The seed value to set for reproducibility.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For PyTorch 1.8 and newer, use this to ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
