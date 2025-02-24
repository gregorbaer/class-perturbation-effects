from typing import Optional, Tuple, Union

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, TensorDataset

from tsxai.utils.logging import setup_logger

logger = setup_logger(__name__)


def prepare_dataloaders(
    data,
    batch_size: int,
    val_split: Optional[float] = None,
    stratify: bool = True,
    seed: int = 42,
) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
    """Prepares PyTorch DataLoaders for training, validation, and testing.

    Creates DataLoader objects with appropriate batching and splitting logic.
    Handles optional validation set creation with stratified or random splitting.

    Args:
        data: Object containing X_train, y_train, X_test, y_test numpy arrays.
        batch_size: Number of samples per batch.
        val_split: If provided, fraction of training data to use for validation.
            Defaults to None.
        stratify: Whether to use stratified split for validation data.
            Defaults to True.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        If val_split is None:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
        If val_split is provided:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
    """
    logger.info(
        f"Preparing dataloaders (batch_size={batch_size}, val_split={val_split})"
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,  # Intentionally 0 - multiple workers slower for in-memory dataset
        "pin_memory": torch.cuda.is_available(),  # only pin if GPU available
    }

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create training dataset
    original_train_dataset = TensorDataset(
        torch.from_numpy(data.X_train.astype(np.float32)),
        torch.from_numpy(data.y_train.astype(np.int32)),
    )

    # Create test dataset
    test_dataset = TensorDataset(
        torch.from_numpy(data.X_test.astype(np.float32)),
        torch.from_numpy(data.y_test.astype(np.int32)),
    )

    logger.debug(
        f"Dataset sizes - Train: {len(original_train_dataset)}, Test: {len(test_dataset)}"
    )
    # Create validation split if requested
    if val_split is not None:
        if stratify:
            logger.debug("Creating stratified validation split")
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=val_split, random_state=seed
            )
            # Get indices for split
            train_idx, val_idx = next(
                splitter.split(
                    data.X_train,
                    data.y_train.argmax(axis=1)
                    if len(data.y_train.shape) > 1
                    else data.y_train,
                )
            )
            # Create subset datasets from original dataset
            train_dataset = Subset(original_train_dataset, train_idx)
            val_dataset = Subset(original_train_dataset, val_idx)
        else:
            logger.debug("Creating random validation split")
            # Calculate split sizes
            val_size = int(len(original_train_dataset) * val_split)
            train_size = len(original_train_dataset) - val_size
            # Random split
            train_dataset, val_dataset = torch.utils.data.random_split(
                original_train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(seed),
            )

        # Create dataloaders
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

        logger.info(
            f"Created dataloaders - Train: {len(train_dataset)}, "
            f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )
        return train_loader, val_loader, test_loader

    # Without validation split
    train_loader = DataLoader(original_train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    logger.info(
        f"Created dataloaders - Train: {len(original_train_dataset)}, "
        f"Test: {len(test_dataset)}"
    )
    return train_loader, test_loader
