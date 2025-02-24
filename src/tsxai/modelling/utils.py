from pathlib import Path

import torch

from tsxai.data.loading import UCRDataset
from tsxai.modelling.models import InceptionTime, ResNet
from tsxai.utils.logging import setup_logger

logger = setup_logger(__name__)


def load_trained_model(
    data: UCRDataset, dataset_name: str, model_name: str, model_dir: str
) -> torch.nn.Module:
    """Load a trained pytorch lightning model for a specific UCR dataset.

    Handles loading on both CPU and GPU devices automatically. If CUDA is not available,
    models will be loaded on CPU.

    Args:
        data (UCRDataset): Dataset object containing the data.
        dataset_name (str): Name of the UCR dataset the model was trained on.
        model_name (str): Name of the model architecture ('ResNet' or 'InceptionTime').
        model_dir (str): Directory containing the saved model weights.

    Returns:
        torch.nn.Module: Loaded model with trained weights.

    Raises:
        ValueError: If model_name is not recognized.
        FileNotFoundError: If model weights file is not found.
    """
    logger.info(f"Loading trained {model_name} model for dataset {dataset_name}")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device} for loading weights")

    # Initialize model architecture
    c_in, c_out = data.n_features, data.n_classes
    if model_name == "ResNet":
        model = ResNet(c_in, c_out)
    elif model_name == "InceptionTime":
        model = InceptionTime(c_in, c_out)
    else:
        logger.error(f"Failed to initialize model: unknown architecture {model_name}")
        raise ValueError(f"Unknown model architecture: {model_name}")

    logger.debug(
        f"Initialized {model_name} with {c_in} input channels and {c_out} classes"
    )

    # Load model weights
    weights_path = Path(model_dir) / dataset_name / f"{model_name}.ckpt"
    if not weights_path.exists():
        logger.error(f"Model weights not found at {weights_path}")
        raise FileNotFoundError(f"No weights file found at {weights_path}")

    # Load weights from Lightning checkpoint with appropriate device mapping
    logger.debug(f"Loading weights from {weights_path}")
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint["state_dict"]

        # Remove the 'model.' prefix from state dict keys
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        # Load the modified state dict and set model to evaluation mode
        model.load_state_dict(new_state_dict)
        model = model.to(device)  # Move model to appropriate device
        model.eval()

        logger.info(
            f"Successfully loaded trained {model_name} model for {dataset_name} on {device}"
        )
        return model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")
