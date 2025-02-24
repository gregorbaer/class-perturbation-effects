from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lets_plot import (
    aes,
    element_rect,
    geom_area,
    geom_line,
    geom_ribbon,
    ggplot,
    ggsize,
    ggtitle,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_minimal2,
)
from tqdm import tqdm

from tsxai.attribution.evaluation.perturbation import PERTURBATION_FUNCTIONS

# Valid AUC type literals
AUCType = Literal["MoRF", "LeRF"]

# Define valid label selection options
LabelType = Union[Literal["predicted", "true"], int]


@dataclass
class PerturbationResult:
    """Results from perturbation-based attribution analysis.

    Attributes:
        scores: Dict mapping AUC type to list of prediction scores at each perturbation step
        auc_scores: Dict mapping AUC type to AUC score
    """

    scores: Dict[str, List[float]]
    auc_scores: Dict[str, float]

    def calculate_ds(self) -> float:
        """Calculate Degradation Score (DS) between MoRF and LeRF curves.

        The DS score measures how much better the MoRF curve degrades predictions
        compared to LeRF. Higher values indicate better attribution quality.
        It's calculated as the sum of differences between MoRF and LeRF scores.

        Returns:
            float: DS score. Higher values indicate larger difference between
                  MoRF and LeRF curves (better attribution quality).

        Raises:
            ValueError: If either MoRF or LeRF scores are missing
        """
        if "MoRF" not in self.scores or "LeRF" not in self.scores:
            raise ValueError("DS score requires both MoRF and LeRF curves")

        morf_scores = np.array(self.scores["MoRF"])
        lerf_scores = np.array(self.scores["LeRF"])
        return np.sum(lerf_scores - morf_scores) / len(morf_scores)

    def calculate_dds(self) -> float:
        """Calculate Decaying Degradation Score (DDS) between MoRF and LeRF curves.

        DDS extends the DS score by adding cubic weighting that emphasizes the initial
        perturbation steps.         The score is calculated as:
            DDS = Σ (Lᵢ - Mᵢ) * ((n-i+1)/n)³  [/ Σ((n-i+1)/n)³ if normalized]
        where:
            Lᵢ, Mᵢ = LeRF and MoRF scores at step i
            n = total number of perturbation steps

        This weighting scheme puts more emphasis on the differences in the early
        perturbation steps, which are typically more important for attribution quality.

        Returns:
            float: DDS score. Higher values indicate better attribution quality,
                with extra emphasis on early perturbation differences. When
                normalized, values are approximately in [-1, 1] range.

        Raises:
            ValueError: If either MoRF or LeRF scores are missing
        """
        if "MoRF" not in self.scores or "LeRF" not in self.scores:
            raise ValueError("DDS score requires both MoRF and LeRF curves")

        morf_scores = np.array(self.scores["MoRF"])
        lerf_scores = np.array(self.scores["LeRF"])
        n = len(morf_scores)

        # Calculate position-based weights with cubic decay
        positions = np.arange(n, 0, -1)
        weights = (positions / n) ** 3

        # Calculate weighted differences
        differences = lerf_scores - morf_scores

        # Calculate weighted sum
        dds = np.sum(differences * weights) / np.sum(weights)

        return dds


class PerturbationEvaluator:
    """Analyzes feature attributions by incrementally perturbing input data
    according to attribution importance and measuring prediction changes.
    """

    VALID_AUC_TYPES = {"MoRF", "LeRF"}

    def __init__(
        self,
        features_per_step: int = 1,
        auc_types: Optional[List[AUCType]] = ["MoRF", "LeRF"],
        perturbation_ratio: float = 1.0,
    ):
        """Initialize the perturbation analysis.

        Args:
            features_per_step: Number of features to perturb in each iteration.
            auc_types: List of AUC types to compute.
                Valid values are "MoRF" (Most Relevant Features) and/or "LeRF" (Least Relevant Features).
                Default is to compute both as they provide complementary information.
            perturbation_ratio: Ratio of features to perturb (0.0 to 1.0).
                For MoRF, perturbs the most important X fraction of features.
                For LeRF, perturbs the least important X fraction of features.
                Default is 1.0 (perturb all features).

        Raises:
            ValueError: If features_per_step < 1, invalid AUC type provided, or
                      perturbation_ratio not in [0, 1]
        """
        if features_per_step < 1:
            raise ValueError("features_per_step must be at least 1")
        if not 0.0 <= perturbation_ratio <= 1.0:
            raise ValueError("perturbation_ratio must be between 0.0 and 1.0")

        self.features_per_step = features_per_step
        self.perturbation_ratio = perturbation_ratio
        self.auc_types = auc_types or ["MoRF"]

        invalid_types = set(self.auc_types) - self.VALID_AUC_TYPES
        if invalid_types:
            raise ValueError(
                f"Invalid AUC types: {invalid_types}. "
                f"Valid options are: {self.VALID_AUC_TYPES}"
            )

    def _apply_perturbation_strategy(
        self,
        x: np.ndarray,
        strategy: Union[str, float, int],
        k: Optional[Union[int, float]] = None,
    ) -> Union[float, np.ndarray]:
        """Apply the selected perturbation strategy to the input data."""
        if isinstance(strategy, (int, float)):
            return float(strategy)

        if strategy not in PERTURBATION_FUNCTIONS:
            raise ValueError(f"Invalid perturbation strategy: {strategy}")

        func = PERTURBATION_FUNCTIONS[strategy]
        if strategy in ["subsequence_mean", "swap"]:
            if k is None:
                raise ValueError(f"k parameter required for {strategy}")
            return func(x[0], k)
        return func(x[0])

    def _get_importance_ordered_indices(
        self, attributions: np.ndarray, auc_type: str
    ) -> List[Tuple[int, int]]:
        """Generate sequence of (feature_idx, time_idx) tuples ordered by importance."""
        n_features, n_timesteps = attributions.shape
        total_elements = n_features * n_timesteps
        n_elements_to_perturb = int(np.ceil(total_elements * self.perturbation_ratio))

        # Get flattened importance order
        importance = np.abs(attributions).reshape(-1)
        ordered_indices = np.argsort(importance)
        if auc_type == "MoRF":
            ordered_indices = ordered_indices[::-1]

        # Convert to feature, time index pairs
        indices = []
        for idx in ordered_indices[:n_elements_to_perturb]:
            feature_idx = idx // n_timesteps
            time_idx = idx % n_timesteps
            indices.append((feature_idx, time_idx))

        return indices

    def evaluate(
        self,
        explainer: "TimeSeriesExplainer",
        sample_index: int,
        attributions: np.ndarray,
        perturbation_strategy: Union[str, float, int],
        k: Optional[Union[int, float]] = None,
        target_class: Optional[int] = None,
    ) -> PerturbationResult:
        """Evaluate attribution importance through sequential perturbation."""
        # Initial validation and setup
        x = explainer.data[sample_index : sample_index + 1]
        pred = explainer._compute_predictions(sample_index)
        target_class = target_class if target_class is not None else pred.label

        if attributions.shape != x.shape[1:]:
            raise ValueError(
                f"Attribution shape {attributions.shape} mismatches input {x.shape[1:]}"
            )

        # Calculate perturbation values
        perturbation = self._apply_perturbation_strategy(x, perturbation_strategy, k)

        results = {
            auc_type: [float(pred.all_probabilities[target_class])]
            for auc_type in self.auc_types
        }

        # Process each AUC type
        for auc_type in self.auc_types:
            ordered_indices = self._get_importance_ordered_indices(
                attributions, auc_type
            )
            x_perturbed = x.copy()

            # Apply sequential perturbation
            for step in range(0, len(ordered_indices), self.features_per_step):
                step_indices = ordered_indices[step : step + self.features_per_step]

                # Apply perturbation for current step
                for feature_idx, time_idx in step_indices:
                    if isinstance(perturbation, (float, int)):
                        x_perturbed[0, feature_idx, time_idx] = perturbation
                    else:
                        x_perturbed[0, feature_idx, time_idx] = perturbation[
                            feature_idx, time_idx
                        ]

                # Compute new prediction
                orig_data = explainer.data[sample_index].copy()
                explainer.data[sample_index] = x_perturbed[0]
                pred = explainer._compute_predictions(sample_index)
                results[auc_type].append(float(pred.all_probabilities[target_class]))
                explainer.data[sample_index] = orig_data

        # Calculate AUC scores
        auc_scores = {
            auc_type: self.calculate_auc(scores) for auc_type, scores in results.items()
        }

        return PerturbationResult(scores=results, auc_scores=auc_scores)

    def calculate_auc(self, scores: List[float]) -> float:
        """Calculate normalized area under the curve score."""
        if len(scores) < 2:
            return 0.0

        x = np.linspace(0, 1, len(scores))
        y = np.array(scores)
        y_range = y.max() - y.min()

        if y_range > 0:
            y = (y - y.min()) / y_range

        return np.trapz(y, x)


def plot_perturbation_analysis_curves(
    results: PerturbationResult,
    perturbation_ratio: float,
    title: str = "Perturbation Analysis Curves",
    plot_size: tuple = (800, 500),
    include_ds: bool = True,
    include_dds: bool = True,
    plot_between: bool = False,
) -> ggplot:
    """Plot perturbation curves with accurate feature perturbation percentage scaling.

    Args:
        results: Perturbation analysis results
        perturbation_ratio: Ratio of features perturbed (0.0 to 1.0)
        title: Title for the plot
        plot_size: Plot size (width, height) in pixels
        include_ds: Whether to include DS score in subtitle
        include_dds: Whether to include DDS scores in subtitle
        plot_between: If True, highlights area between curves in gray

    Returns:
        lets-plot visualization object
    """
    # Check available curves and provide warnings if needed
    available_curves = list(results.scores.keys())
    both_curves_available = "MoRF" in available_curves and "LeRF" in available_curves

    if plot_between and not both_curves_available:
        print(
            "Warning: Cannot plot area between curves - LeRF or MoRF missing. "
            "Falling back to individual areas."
        )
        plot_between = False

    # Create DataFrame for plotting with scaled percentages
    plot_data = []
    for auc_type, scores in results.scores.items():
        if not scores:
            print(f"Warning: No scores available for {auc_type}")
            continue

        # Scale percentages based on perturbation ratio
        max_perturbation = 100 * perturbation_ratio
        df = pd.DataFrame(
            {
                "perturbed_pct": np.linspace(0, max_perturbation, len(scores)),
                "score": scores,
                "type": auc_type,
            }
        )
        plot_data.append(df)

    if not plot_data:
        raise ValueError("No valid perturbation scores available for plotting")

    data = pd.concat(plot_data, ignore_index=True)

    # Create subtitle with scores
    auc_parts = []
    for auc_type, score in results.auc_scores.items():
        if np.isnan(score):
            print(f"Warning: AUC score for {auc_type} is NaN")
            continue
        auc_parts.append(f"{auc_type}: {score:.3f}")

    score_parts = [f"AUC ({' | '.join(auc_parts)})"] if auc_parts else []

    # Add DS score if requested and possible
    if include_ds and both_curves_available:
        try:
            ds_score = results.calculate_ds()
            if not np.isnan(ds_score):
                score_parts.append(f"DS: {ds_score:.3f}")
        except Exception as e:
            print(f"Warning: Could not calculate DS score: {str(e)}")
    elif include_ds:
        print("Warning: Cannot calculate DS score - requires both MoRF and LeRF curves")

    # Add DDS score if requested and possible
    if include_dds and both_curves_available:
        try:
            dds_score = results.calculate_dds()
            if not np.isnan(dds_score):
                score_parts.append(f"DDS: {dds_score:.3f}")
        except Exception as e:
            print(f"Warning: Could not calculate DDS score: {str(e)}")
    elif include_dds:
        print(
            "Warning: Cannot calculate DDS score - requires both MoRF and LeRF curves"
        )

    subtitle = " | ".join(score_parts) if score_parts else None

    NATURE_BINARY = [
        "#006699",  # Nature Blue
        "#DC2830",  # Nature Red
    ]
    # Get curve types in consistent order
    available_curves = list(results.scores.keys())
    curve_types = []
    if "MoRF" in available_curves:
        curve_types.append("MoRF")
    if "LeRF" in available_curves:
        curve_types.append("LeRF")

    # Create color mappings that align with curve types
    colors = {curve_type: NATURE_BINARY[i] for i, curve_type in enumerate(curve_types)}
    color_values = [colors[curve_type] for curve_type in curve_types]

    # Modify x-axis scale based on perturbation ratio
    max_pct = 100 * perturbation_ratio
    x_breaks = np.linspace(0, max_pct, min(6, int(max_pct / 10) + 1))

    # Create base plot
    plot = ggplot()

    if plot_between and both_curves_available:
        # Create data for area between curves with correct scaling
        morf_len = len(results.scores["MoRF"])
        between_data = pd.DataFrame(
            {
                "perturbed_pct": np.linspace(0, max_pct, morf_len),
                "score_morf": results.scores["MoRF"],
                "score_lerf": results.scores["LeRF"],
            }
        )
        plot = plot + geom_ribbon(
            data=between_data,
            mapping=aes(x="perturbed_pct", ymin="score_morf", ymax="score_lerf"),
            fill="gray",
            alpha=0.3,
        )
    if not plot_between:
        plot = plot + geom_area(
            data=data,
            mapping=aes(x="perturbed_pct", y="score", fill="type"),
            alpha=0.2,
            position="identity",
        )

    # Add line traces on top
    plot = (
        plot
        + geom_line(
            data=data,
            mapping=aes(x="perturbed_pct", y="score", color="type"),
            size=1.25,
        )
        + scale_x_continuous(
            name="Percentage of Features Perturbed (%)", breaks=list(x_breaks)
        )
        + scale_y_continuous(
            name="Target Class Probability", limits=[0, data["score"].max() * 1.05]
        )
        + scale_color_manual(values=color_values)
        + scale_fill_manual(values=color_values)
        + ggtitle(title, subtitle=subtitle)
        + theme_minimal2()
        + theme(
            legend_position="bottom",
            legend_direction="horizontal",
            legend_background=element_rect(color="black", size=0.25),
        )
        + ggsize(plot_size[0], plot_size[1])
        + labs(color="Perturbation Type", fill="Perturbation Type")
    )

    return plot


def evaluate_correctness_for_samples(
    explainer: "TimeSeriesExplainer",
    sample_ids: list[int],
    method: str,
    perturbation_strategy: str,
    label: Union[str, int],
    features_per_step: float,
    perturbation_ratio: float,
    rescale: bool = False,
) -> pd.DataFrame:
    """Evaluate attribution correctness metrics for multiple samples.

    Args:
        explainer: TimeSeriesExplainer instance
        sample_ids: List of sample indices to evaluate
        method: Attribution method to use
        perturbation_strategy: Strategy for perturbing features (e.g., "zero", "mean")
        label: Label to explain - can be "predicted", "true", or specific class index
        features_per_step: Number of features to perturb
        perturbation_ratio: Maximum ratio of features to perturb
        rescale: Whether to rescale attributions

    Returns:
        pd.DataFrame: Results containing sample information and metrics:
            - idx: Sample index
            - true_label: True class label
            - predicted_label: Model's predicted label
            - probabilities: Model's prediction probabilities
            - auc_morf: Area under curve for Most Relevant First
            - auc_lerf: Area under curve for Least Relevant First
            - ds: Degradation Score
            - dds: Decaying Degradation Score (unnormalized)
    """
    results = {
        "idx": [],
        "true_label": [],
        "predicted_label": [],
        "probabilities": [],
        "auc_morf": [],
        "auc_lerf": [],
        "ds": [],
    }

    # Initialize evaluator once for all samples
    evaluator = PerturbationEvaluator(
        features_per_step=features_per_step,
        auc_types=["MoRF", "LeRF"],
        perturbation_ratio=perturbation_ratio,
    )

    for sample_idx in tqdm(sample_ids):
        # Explain the sample
        explanation = explainer.explain(
            sample_index=sample_idx, method=method, label=label, rescale=rescale
        )

        # Determine which label to evaluate
        if label == "predicted":
            label_to_eval = explanation.prediction.label
        elif label == "true":
            label_to_eval = explanation.metadata.true_label
        else:
            label_to_eval = label

        # Compute evaluation metrics
        try:
            result = evaluator.evaluate(
                explainer=explainer,
                sample_index=sample_idx,
                attributions=explanation.data.attributions,
                target_class=label_to_eval,
                perturbation_strategy=perturbation_strategy,
                k=0.1
                if perturbation_strategy in ["subsequence_mean", "swap"]
                else None,
            )

            # Store basic information
            results["idx"].append(sample_idx)
            results["true_label"].append(explanation.metadata.true_label)
            results["predicted_label"].append(explanation.prediction.label)
            results["probabilities"].append(explanation.prediction.all_probabilities)

            # Store metrics
            results["auc_morf"].append(result.auc_scores["MoRF"])
            results["auc_lerf"].append(result.auc_scores["LeRF"])
            results["ds"].append(result.calculate_ds())

        except Exception as e:
            print(f"Failed for sample {sample_idx}: {str(e)}")
            # Store NaN values for failed evaluations
            results["idx"].append(sample_idx)
            results["true_label"].append(explanation.metadata.true_label)
            results["predicted_label"].append(explanation.prediction.label)
            results["probabilities"].append(explanation.prediction.all_probabilities)
            results["auc_morf"].append(np.nan)
            results["auc_lerf"].append(np.nan)
            results["ds"].append(np.nan)

    return pd.DataFrame(results)


def calculate_class_adjusted_metric(
    df: pd.DataFrame,
    metric_col: str,
    group_cols: List[str],
    class_col: str,
    alpha: float,
) -> pd.DataFrame:
    """Calculate aggregate class-adjusted metrics for groups in a DataFrame.

    Computes an adjusted score that combines the overall mean with a penalty term for
    class-specific differences:
        adjusted_score = mean(metric) - α * Δ
    where Δ is the normalized mean absolute difference between class-specific means.
    For binary classification, Δ = |mean_1 - mean_0|/2.
    For multiclass, Δ = mean(|mean_i - mean_j|)/2 for all class pairs i,j.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data. Expects long format.
        metric_col (str): Name of the column containing the metric values.
        group_cols (List[str]): Columns to group by before calculating metrics.
        class_col (str): Name of the column containing class labels.
        alpha (float): Weight of the class difference penalty (typically in [0,1]).
            Setting α=0 ignores class differences, while α=1 gives equal weight to
            average performance and average differences between classes.

    Returns:
        pd.DataFrame: DataFrame with group columns and calculated adjusted metric.

    Example:
        For a degradation score (DS) in [-1,1], with α=1:
        - If all classes have similar DS → penalty ≈ 0 → adjusted ≈ mean(DS)
        - If classes have different DS → penalty > 0 → adjusted < mean(DS)
    """

    def calc_group_metric(group):
        # Calculate average metric
        avg_metric = group[metric_col].mean()

        # Calculate class differences
        class_means = group.groupby(class_col)[metric_col].mean()
        if len(class_means) == 2:
            class_diff = abs(class_means.iloc[1] - class_means.iloc[0])
        else:
            pairs = list(combinations(class_means.index, 2))
            diffs = [abs(class_means[i] - class_means[j]) for i, j in pairs]
            class_diff = np.mean(diffs)

        # Calculate adjusted score
        return avg_metric - alpha * (class_diff / 2)

    return (
        df.groupby(group_cols, group_keys=True)
        .apply(calc_group_metric)
        .reset_index(name="adjusted_metric")
    )
