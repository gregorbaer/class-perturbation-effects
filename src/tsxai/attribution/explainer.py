from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lets_plot import ggplot
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR

from tsxai.data.loading import UCRDataset
from tsxai.utils.logging import setup_logger
from tsxai.visualization.visualize import plot_ts_attributions

logger = setup_logger(__name__)

# Custom types for clarity
Label = Union[Literal["predicted", "true"], int]
TimeSeriesArray = np.ndarray  # Shape: (n_samples, n_timesteps, n_features)
AttributionArray = np.ndarray  # Shape: (n_features, n_timesteps)


@dataclass
class Prediction:
    """Container for model prediction results."""

    label: int
    probability: float
    all_probabilities: np.ndarray
    explained_label: int


@dataclass
class Metadata:
    """Container for explanation metadata."""

    true_label: int
    observation_index: int


@dataclass
class Data:
    """Container for time series data and attributions."""

    ds: np.ndarray
    y: np.ndarray
    attributions: np.ndarray
    method: Optional[str]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the data to a pandas DataFrame format suitable for visualization."""
        n_features = self.y.shape[0] if self.y.ndim == 2 else 1

        if n_features > 1:  # Multiple features
            dfs = []
            for i in range(n_features):
                df = pd.DataFrame(
                    {
                        "ds": self.ds,
                        "y": self.y[i],
                        "r": self.attributions[i],
                        "feature": f"Feature {i}",
                    }
                )
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        else:  # Single feature
            return pd.DataFrame(
                {
                    "ds": self.ds,
                    "y": self.y.flatten() if self.y.ndim == 2 else self.y,
                    "r": self.attributions.flatten()
                    if self.attributions.ndim == 2
                    else self.attributions,
                }
            )


@dataclass
class ExplanationResult:
    """Container for complete explanation results."""

    metadata: Metadata
    prediction: Prediction
    data: Data

    def visualize_attributions(
        self,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        plot_size: Tuple[int, int] = (800, 350),
        hm_theme: str = "inferno",
        ts_color: str = "white",
        ts_size: int = 2,
        y_lim: Optional[Tuple[float, float]] = None,
        scale_fill_limits: Optional[Tuple[float, float]] = None,
    ) -> ggplot:
        """
        Visualize the time series data with attribution overlay.

        Parameters:
        -----------
        title : Optional[str]
            Custom title for the plot. If None, a default title is generated.
        subtitle : Optional[str]
            Custom subtitle for the plot
        plot_size : Tuple[int, int]
            Size of the plot in pixels (width, height)
        hm_theme : str
            Color theme for the heatmap (viridis colormap option)
        ts_color : str
            Color of the time series line
        ts_size : int
            Size of the time series line
        y_lim : Optional[Tuple[float, float]]
            Custom y-axis limits. If None, automatically determined.
        scale_fill_limits : Optional[Tuple[float, float]]
            Custom limits for attribution color scale. If None, uses [0, 1].

        Returns:
        --------
        plotnine.ggplot
            The generated plot
        """
        # Convert data to DataFrame
        df = self.data.to_dataframe()

        # Generate default title if none provided
        all_probs = self.prediction.all_probabilities

        probs_str = ", ".join([f"P({i})={p:.3f}" for i, p in enumerate(all_probs)])
        probs_str = f"[{probs_str}]"

        if title is None and subtitle is None:
            title = (
                f"Attributions ({self.data.method}) "
                + f"for observation {self.metadata.observation_index}"
            )
            is_prediction_true = (
                "Correct"
                if self.prediction.label == self.metadata.true_label
                else "False"
            )
            subtitle = (
                f"Explaining label {self.prediction.explained_label} "
                + probs_str
                + f" (Prediction={is_prediction_true})"
            )

        # Only use faceting for multiple features
        n_features = self.data.y.shape[0] if self.data.y.ndim == 2 else 1
        facet_col = "feature" if n_features > 1 else None

        # Call the plotting function with our pre-set parameters
        return plot_ts_attributions(
            ts_df=df,
            x_col="ds",
            y_col="y",
            exp="r",
            facet_col=facet_col,
            title=title,
            subtitle=subtitle,
            plot_size=plot_size,
            hm_theme=hm_theme,
            ts_color=ts_color,
            ts_size=ts_size,
            y_lim=y_lim,
            scale_fill_limits=scale_fill_limits or [0, 1],
            exp_label=f"Attribution ({self.data.method})",
            x_lab="Time Step",
            y_lab="Value",
        )


class TimeSeriesExplainer:
    """Class for explaining time series predictions using various attribution methods."""

    def __init__(
        self,
        dataset: UCRDataset,
        model: torch.nn.Module,
        split: Literal["train", "test"] = "test",
        device: str = "cpu",
        seed: int = 42,
    ):
        """Initialize the TimeSeriesExplainer.

        Sets up the explainer with the provided dataset, model, and configuration
        for generating time series explanations.

        Args:
            dataset: Dataset containing both features and labels.
            model: The trained model to explain.
            split: Which dataset split to use for explanations. Defaults to "test".
            device: Device to use for computations ("cpu" or "cuda").
                Only supports "cpu" currently. Defaults to "cpu".
            seed: Random seed for reproducibility. Defaults to 42.

        Raises:
            ValueError: If input shapes are incompatible or invalid.
            TypeError: If inputs are of wrong type.
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")

        # Select appropriate data split
        self.data = dataset.X_train if split == "train" else dataset.X_test
        self.labels = dataset.y_train if split == "train" else dataset.y_test

        # Store model and parameters
        self.model = model
        self.device = device
        self.seed = seed
        self.n_timesteps = dataset.n_timesteps
        self.n_features = dataset.n_features
        self.n_classes = dataset.n_classes

        # Basic validation
        if self.data.ndim != 3:
            raise ValueError(
                f"Input data must be 3-dimensional (n_samples, n_features, n_timesteps), got shape {self.data.shape}"
            )

        logger.debug(
            f"Initialized TimeSeriesExplainer with {len(self.data)} samples and {self.n_classes} classes"
        )

    def _compute_predictions(self, sample_index: int) -> Prediction:
        """Compute model predictions for a specific observation."""
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            # Data is already in (n_samples, n_features, n_timesteps) format
            input_tensor = (
                torch.from_numpy(self.data[sample_index : sample_index + 1])
                .to(self.device)
                .float()
            )

            output = self.model(input_tensor)
            if output.shape[1] == 1:  # Binary classification
                probabilities = torch.sigmoid(output).cpu().numpy()[0]
                probabilities = np.array(
                    [1 - probabilities[0], probabilities[0]]
                )  # Convert to [neg_prob, pos_prob]
            else:  # Multiclass classification
                probabilities = F.softmax(output, dim=1).cpu().numpy()[0]

            predicted_label = int(np.argmax(probabilities))
            probability = float(probabilities[predicted_label])

        logger.debug(
            f"Predicted label {predicted_label} with probability {probability:.4f}"
        )

        return Prediction(
            label=predicted_label,
            probability=probability,
            all_probabilities=probabilities,
            explained_label=predicted_label,  # Will be updated if needed
        )

    def _get_explanation_label(
        self, sample_index: int, label: Label, prediction: Prediction
    ) -> int:
        """Determine which label to use for generating explanations."""
        if isinstance(label, str):
            if label == "predicted":
                return prediction.label
            elif label == "true":
                return int(self.labels[sample_index])
            else:
                raise ValueError('String label must be either "predicted" or "true"')
        elif isinstance(label, int):
            if 0 <= label < self.n_classes:
                return label
            else:
                raise ValueError(
                    f"Label index {label} out of range [0, {self.n_classes - 1}]"
                )
        else:
            raise TypeError(
                "Label must be either string ('predicted'/'true') or integer"
            )

    def _compute_attributions(
        self,
        sample_index: int,
        explanation_label: int,
        method: str = "FA",
        rescale: Optional[bool] = False,
        mode: str = "feat",
    ) -> AttributionArray:
        """Compute attributions for a specific observation."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.model.to(self.device)

        explanation_model = TSR(
            model=self.model,
            NumTimeSteps=self.n_timesteps,
            NumFeatures=self.n_features,
            method=method,
            mode=mode,
            device=self.device,
        )

        observation = self.data[sample_index : sample_index + 1]

        attributions = explanation_model.explain(
            item=observation,
            labels=explanation_label,
            TSR=False if rescale is None else rescale,
            attribution=0.0,
        )

        logger.debug(
            f"Computed attributions with shape {attributions.shape} for label {explanation_label}"
        )
        return attributions

    def _prepare_time_series_data(
        self,
        sample_index: int,
        attributions: AttributionArray,
        method: Optional[str] = None,
    ) -> Data:
        """Prepare time series data and attributions for output."""
        time_steps = np.arange(self.n_timesteps)
        feature_values = self.data[sample_index]

        return Data(
            ds=time_steps, y=feature_values, attributions=attributions, method=method
        )

    def explain(
        self,
        sample_index: int,
        method: str,
        rescale: Optional[bool] = False,
        mode: str = "feat",
        label: Label = "predicted",
    ) -> ExplanationResult:
        """Generate explanation for a specific observation.

        This method produces attributions for the specified sample using the selected
        explanation method and configuration.

        Args:
            sample_index: Index of the observation to explain.
            method: Attribution method to use.
            rescale: Whether to rescale the attributions. Defaults to False.
            mode: Mode of explanation. Defaults to "feat".
            label: Label to explain: "predicted", "true", or specific class index.
                Defaults to "predicted".

        Returns:
            Structured explanation result containing metadata, prediction information,
            and attribution data.

        Raises:
            ValueError: If sample_index is out of range or parameters are invalid.
        """
        if not 0 <= sample_index < len(self.data):
            raise ValueError(
                f"sample_index {sample_index} out of range [0, {len(self.data) - 1}]"
            )

        logger.debug(
            f"Generating explanation for observation {sample_index} with label type '{label}'"
        )

        # Get true label and predictions
        true_label = int(self.labels[sample_index])
        prediction = self._compute_predictions(sample_index)

        # Determine which label to explain
        explanation_label = self._get_explanation_label(sample_index, label, prediction)
        logger.debug(
            f"Explaining label {explanation_label} (true: {true_label}, predicted: {prediction.label})"
        )

        # Update prediction with explained label
        prediction.explained_label = explanation_label

        # Compute attributions
        attributions = self._compute_attributions(
            sample_index, explanation_label, method, rescale, mode
        )

        # Prepare data
        data = self._prepare_time_series_data(sample_index, attributions, method)

        # Create and return result
        return ExplanationResult(
            metadata=Metadata(true_label=true_label, observation_index=sample_index),
            prediction=prediction,
            data=data,
        )
