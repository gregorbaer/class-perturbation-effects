from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate
from tsai.data.external import get_UCR_data, get_UCR_univariate_list

from tsxai.utils.logging import setup_logger

# Create logger for this module
logger = setup_logger(__name__)


@dataclass
class UCRDataset:
    """Container for UCR time series dataset with preprocessed data and metadata.

    This class holds both the input time series data (X) and the corresponding labels (y)
    for both training and test sets. By default, labels are provided as integer values
    in range [0, n_classes-1]. Optionally, one-hot encoded labels can be included if
    requested during loading.

    Attributes:
        X_train: Training time series data of shape (n_samples, n_channels, n_timesteps)
        y_train: Integer training labels of shape (n_samples,), values in [0, n_classes-1]
        X_test: Test time series data of shape (n_samples, n_channels, n_timesteps)
        y_test: Integer test labels of shape (n_samples,), values in [0, n_classes-1]
        y_train_onehot: Optional one-hot encoded training labels (n_samples, n_classes)
        y_test_onehot: Optional one-hot encoded test labels (n_samples, n_classes)
        n_classes: Number of unique classes in the dataset
        n_features: Number of features/channels in the time series data
        n_timesteps: Length of each time series
        class_distribution: Dictionary containing class distribution percentages for each set
            Format: {class_idx: {'train': pct_train, 'test': pct_test}}
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_train_onehot: Optional[np.ndarray] = None
    y_test_onehot: Optional[np.ndarray] = None
    n_classes: int = None
    n_features: int = None
    n_timesteps: int = None
    class_distribution: Dict[int, Dict[str, float]] = None

    def to_plotting_df(self, split: str = "train") -> pd.DataFrame:
        """Convert the dataset to a DataFrame format suitable for visualization.

        Creates a long-format DataFrame where each row represents one timestep of one
        time series. This format is compatible with plotting functions that expect
        columns: unique_id (time series identifier), ds (timestep), y (value),
        and label (class).

        Args:
            split: Which data split to use, either 'train' or 'test'

        Returns:
            DataFrame with columns [unique_id, ds, y, label] where:
                - unique_id: identifier for each time series
                - ds: timestep index
                - y: time series value at that timestep
                - label: integer class label [0, n_classes-1]

        Raises:
            ValueError: If split is not 'train' or 'test'
        """
        if split not in ["train", "test"]:
            raise ValueError("split must be either 'train' or 'test'")

        # Select appropriate data
        X = self.X_train if split == "train" else self.X_test
        y = self.y_train if split == "train" else self.y_test

        # Create unique IDs for each time series
        n_samples = len(X)

        # Handle the data reshaping correctly for univariate time series
        X_reshaped = X.squeeze(axis=1) if X.ndim == 3 else X

        # Create DataFrame
        df = pd.DataFrame(
            {
                "unique_id": np.repeat(
                    [f"ts_{i}" for i in range(n_samples)], self.n_timesteps
                ),
                "y": X_reshaped.reshape(-1),
                "ds": np.tile(np.arange(self.n_timesteps), n_samples),
                "label": np.repeat(y, self.n_timesteps),
            }
        )

        return df


def remap_labels(y: np.ndarray, ascending: bool = True) -> np.ndarray:
    """Remap label values to a zero-based consecutive integer range.

    Args:
        y: Array of labels to be remapped
        ascending: If True, lowest original value maps to 0. If False, highest maps to 0.

    Returns:
        Array with same shape as input, but with values remapped to range [0, n_classes-1]

    Example:
        >>> labels = np.array([10, 20, 10, 30, 20])
        >>> remap_labels(labels, ascending=True)
        array([0, 1, 0, 2, 1])
        >>> remap_labels(labels, ascending=False)
        array([2, 1, 2, 0, 1])
    """
    unique_values = np.sort(np.unique(y))
    if not ascending:
        unique_values = unique_values[::-1]
    value_to_index = {value: index for index, value in enumerate(unique_values)}
    return np.array([value_to_index[value] for value in y])


def load_ucr_data(
    dataset_name: str,
    one_hot_encode: bool = False,
    remap_labels_ascending: bool = True,
    on_disk: bool = True,
    local_data_dir: str = "data/UCR",
) -> UCRDataset:
    """Load and preprocess a UCR time series classification dataset.

    This function loads a dataset from the UCR time series classification archive
    and performs the following preprocessing steps:
    1. Remaps class labels to consecutive integers [0, n_classes-1]
    2. Optionally converts labels to one-hot encoding
    3. Calculates class distribution statistics

    Args:
        dataset_name: Name of the UCR dataset to load
        one_hot_encode: Whether to convert integer labels to one-hot encoding
        remap_labels_ascending: If True, lowest original label maps to 0,
            otherwise, highest maps to 0. If None, keep original labels.
        on_disk: Loads the dataset from disk if True and the dataset is present in local_data_dir.
            If not present and on_disk=True, they will be downloaded and saved to local_data_dir.
            If False, the dataset will be loaded from the UCR archive.
        local_data_dir: Directory to store UCR datasets if on_disk is True.

    Returns:
        UCRDataset object containing the processed data and metadata

    Raises:
        ValueError: If dataset not found in UCR archive
        ValueError: If dataset contains NaN or infinite values

    Example:
        >>> data = load_ucr_data("ECG200")
        >>> print(f"Dataset loaded: {data.n_classes} classes, {data.n_timesteps} timesteps")
        >>> train_df = data.to_plotting_df('train')
    """
    logger.info(f"Loading UCR dataset: {dataset_name}")

    # Validate dataset name
    if dataset_name not in get_UCR_univariate_list():
        logger.error(f"Dataset '{dataset_name}' not found in UCR archive")
        raise ValueError(
            f"Dataset '{dataset_name}' not found in UCR univariate datasets. "
            f"Available datasets: {', '.join(get_UCR_univariate_list())}"
        )

    # Load data
    X_train, y_train, X_test, y_test = get_UCR_data(
        dataset_name, on_disk=on_disk, verbose=False, parent_dir=local_data_dir
    )
    logger.info(f"Dataset loaded: train={len(X_train)} test={len(X_test)} samples")

    # Validate data
    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(X_test)):
        logger.error(f"Dataset '{dataset_name}' contains invalid values")
        raise ValueError(f"Dataset '{dataset_name}' contains NaN or infinite values!")

    # Remap labels
    logger.debug("Remapping class labels to consecutive integers")
    if remap_labels_ascending is not None:
        y_train = remap_labels(y_train, ascending=remap_labels_ascending)
        y_test = remap_labels(y_test, ascending=remap_labels_ascending)
    n_classes = len(np.unique(y_train))
    logger.info(f"Found {n_classes} unique classes")

    # Initialize one-hot encoded labels as None
    y_train_onehot, y_test_onehot = None, None

    # One-hot encode if requested
    if one_hot_encode:
        logger.debug("Performing one-hot encoding of labels")
        encoder = OneHotEncoder(sparse=False)
        y_combined = np.concatenate([y_train, y_test]).reshape(-1, 1)
        encoder.fit(y_combined)
        y_train_onehot = encoder.transform(y_train.reshape(-1, 1))
        y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

    # Calculate class distribution using integer labels
    def get_distribution(y: np.ndarray) -> Dict[int, float]:
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts / len(y) * 100))

    train_dist = get_distribution(y_train)
    test_dist = get_distribution(y_test)

    class_distribution = {
        int(label): {"train": train_dist.get(label, 0), "test": test_dist.get(label, 0)}
        for label in set(train_dist.keys()) | set(test_dist.keys())
    }

    # Get the number of features (channels)
    n_features = X_train.shape[1] if X_train.ndim == 3 else 1

    dataset = UCRDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_train_onehot=y_train_onehot,
        y_test_onehot=y_test_onehot,
        n_classes=len(class_distribution),
        n_features=n_features,
        n_timesteps=X_train.shape[-1],
        class_distribution=class_distribution,
    )
    logger.info(
        f"Successfully processed {dataset_name} data: {dataset.n_features} features, "
        f"{dataset.n_timesteps} timesteps"
    )
    return dataset


def print_dataset_info(data: UCRDataset) -> None:
    """Print formatted summary information about the dataset.

    Displays a table with basic dataset statistics (sample counts, dimensions)
    and a detailed breakdown of class distribution in train and test sets.

    Args:
        data: UCRDataset object containing the dataset to summarize

    Example:
        >>> data = load_ucr_data("ECG200")
        >>> print_dataset_info(data)
        ┌────────────────┬───────┐
        │ Metric         │ Value │
        ├────────────────┼───────┤
        │ Train samples  │ 100   │
        │ Test samples   │ 100   │
        │ Num features   │ 1     │
        │ Num classes    │ 2     │
        │ Num timesteps  │ 96    │
        └────────────────┴───────┘
        Class Distribution:
        ┌─────────┬────────┬───────┐
        │ Class   │ Train  │ Test  │
        ├─────────┼────────┼───────┤
        │ Class 0 │ 50.0%  │ 50.0% │
        │ Class 1 │ 50.0%  │ 50.0% │
        └─────────┴────────┴───────┘
    """
    logger.debug("Generating dataset summary")

    # Dataset overview
    overview = [
        ["Train samples", len(data.X_train)],
        ["Test samples", len(data.X_test)],
        ["Num features", data.n_features],
        ["Num classes", data.n_classes],
        ["Num timesteps", data.n_timesteps],
    ]
    print(tabulate(overview, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    # Class distribution
    distribution = [
        [f"Class {cls}", f"{dist['train']:.1f}%", f"{dist['test']:.1f}%"]
        for cls, dist in sorted(data.class_distribution.items())
    ]
    print("\nClass Distribution:")
    print(
        tabulate(
            distribution, headers=["Class", "Train", "Test"], tablefmt="fancy_grid"
        )
    )
