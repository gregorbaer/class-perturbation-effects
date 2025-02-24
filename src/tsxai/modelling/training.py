import os
import random
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBar
from tqdm import tqdm

from tsxai.utils.logging import setup_logger

logger = setup_logger(__name__)


# Default training configuration
DEFAULT_CONFIG = {
    "epochs": 500,  # Maximum number of training epochs
    "patience": 25,  # Early stopping patience
    "learning_rate": 3e-4,  # Initial learning rate
    "weight_decay": 0.001,  # Weight decay for optimizer
    "deterministic": True,  # Enable deterministic training
    "seed": 42,  # Random seed for reproducibility
    "show_progress": True,  # Show progress bar
    "save_model": False,  # Save model checkpoints
    "save_hyperparameters": False,  # Save hyperparameters
    "enable_early_stopping": True,  # Use early stopping
    "scheduler_t0": None,  # Initial cycle length for cosine annealing
    "scheduler_tmult": 2,  # Cycle length multiplier
    "model_dir": "models",  # Directory for model checkpoints
    "log_dir": "logs",  # Directory for logs
}

# Enable tensor cores if available
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability()
    has_tensor_cores = compute_capability[0] >= 7
    if has_tensor_cores:
        torch.set_float32_matmul_precision("medium")
        logger.info("Enabled Tensor Cores with medium precision")


class TimeSeriesLightningModule(pl.LightningModule):
    """PyTorch Lightning module for time series classification.

    Handles training, validation, and optimization logic for time series models.

    Args:
        model: PyTorch model to train
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization factor
        scheduler_t0: Initial cycle length for cosine annealing
        scheduler_tmult: Cycle length multiplier
        save_hyperparameters: Whether to save hyperparameters
    """

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = DEFAULT_CONFIG["learning_rate"],
        weight_decay: float = DEFAULT_CONFIG["weight_decay"],
        scheduler_t0: Optional[int] = DEFAULT_CONFIG["scheduler_t0"],
        scheduler_tmult: int = DEFAULT_CONFIG["scheduler_tmult"],
        save_hyperparameters: bool = DEFAULT_CONFIG["save_hyperparameters"],
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_t0 = scheduler_t0
        self.scheduler_tmult = scheduler_tmult

        if save_hyperparameters:
            self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        """Compute loss for both training and validation."""
        x, y = batch
        output = self(x.float())

        # Determine if this is binary or multiclass based on model output
        if output.shape[1] == 1:  # Binary classification
            loss = F.binary_cross_entropy_with_logits(output.squeeze(-1), y.float())
        else:  # Multiclass classification
            loss = F.cross_entropy(output, y.long())

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        if self.scheduler_t0 is None:
            self.scheduler_t0 = self.trainer.max_epochs // 6

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.scheduler_t0, T_mult=self.scheduler_tmult
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


def setup_deterministic_training(seed: int = 42, warn_only: bool = True) -> None:
    """
    Setup all settings for deterministic training
    """
    # Set CUDA environment variable
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Set all seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enable deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)


class SimpleProgressBar(ProgressBar):
    """A minimal progress bar that shows epoch-level progress."""

    def __init__(self):
        super().__init__()
        self.enable = True
        self.pbar = None

    def on_train_start(self, trainer, pl_module):
        self.pbar = tqdm(
            total=trainer.max_epochs,
            desc="Training Epochs",
            leave=True,
            dynamic_ncols=True,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss", 0)
        val_loss = trainer.callback_metrics.get("val_loss", 0)

        self.pbar.set_postfix(
            {"train_loss": f"{train_loss:.3f}", "val_loss": f"{val_loss:.3f}"}
        )
        self.pbar.update(1)

    def on_train_end(self, trainer, pl_module):
        if self.pbar is not None:
            self.pbar.close()


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
) -> pl.LightningModule:
    """Train a time series classification model using PyTorch Lightning.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary. See DEFAULT_CONFIG for options.
            Unspecified options will use values from DEFAULT_CONFIG.

    Returns:
        Trained PyTorch Lightning module
    """
    # Merge provided config with defaults
    full_config = DEFAULT_CONFIG.copy()
    full_config.update(config)

    # Setup deterministic training if enabled
    if full_config["deterministic"]:
        setup_deterministic_training(seed=full_config["seed"])
        if full_config["show_progress"]:
            logger.info(
                f"Enabled deterministic training with seed {full_config['seed']}"
            )

    if full_config["show_progress"]:
        logger.info(
            f"Training {model.__class__.__name__} on {full_config.get('dataset_name', 'dataset')}"
        )

    # Create Lightning module
    lightning_module = TimeSeriesLightningModule(
        model=model,
        learning_rate=full_config["learning_rate"],
        weight_decay=full_config["weight_decay"],
        scheduler_t0=full_config["scheduler_t0"],
        scheduler_tmult=full_config["scheduler_tmult"],
        save_hyperparameters=full_config["save_hyperparameters"],
    )

    # Configure callbacks
    callbacks = []

    if full_config["enable_early_stopping"]:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=full_config["patience"],
            mode="min",
            verbose=False,
            min_delta=1e-3,  # minimum change threshold
        )
        callbacks.append(early_stopping)

    if full_config["save_model"]:
        # Create full path for model directory
        model_dir = os.path.join(
            full_config["model_dir"], full_config.get("dataset_name", "dataset")
        )
        os.makedirs(model_dir, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=model_dir,
                filename=model.__class__.__name__,
                save_top_k=1,
                mode="min",
                monitor="val_loss",
            )
        )

    if full_config["show_progress"]:
        callbacks.append(SimpleProgressBar())

    # Configure trainer with minimal output
    trainer = pl.Trainer(
        max_epochs=full_config["epochs"],
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        deterministic=full_config["deterministic"],
        precision="16-mixed" if torch.cuda.is_available() else 32,
        enable_progress_bar=True,  # Disable default progress bar
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=full_config["save_model"],
    )

    # Log device information through custom logger
    if full_config["show_progress"]:
        device_info = (
            f"Using {'16bit AMP' if torch.cuda.is_available() else '32bit'} precision | "
            f"GPU: {torch.cuda.is_available()} | "
            f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
        )
        logger.info(device_info)

    # Train the model
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    if full_config["show_progress"]:
        logger.info(
            f"Finished training {model.__class__.__name__} on {full_config.get('dataset_name', 'dataset')}"
        )

    return lightning_module
