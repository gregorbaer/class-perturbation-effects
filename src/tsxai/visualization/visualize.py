from typing import Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tabulate
from lets_plot import (
    aes,
    coord_cartesian,
    element_blank,
    element_rect,
    facet_wrap,
    geom_boxplot,
    geom_hline,
    geom_line,
    geom_point,
    geom_rect,
    geom_text,
    geom_tile,
    geom_violin,
    gggrid,
    ggplot,
    ggsize,
    ggtitle,
    labs,
    layer_tooltips,
    scale_color_manual,
    scale_fill_gradient,
    scale_fill_manual,
    scale_fill_viridis,
    scale_x_discrete,
    scale_y_continuous,
    theme,
    theme_light,
    theme_minimal,
    theme_minimal2,
    xlab,
    ylab,
    ylim,
)
from lets_plot.mapping import as_discrete

from tsxai.attribution.evaluation.metrics import (
    PerturbationEvaluator,
    plot_perturbation_analysis_curves,
)
from tsxai.visualization.utils import ScientificPalette


def sample_ts_per_class(
    df: pd.DataFrame, n_samples=3, random_state: int = None
) -> pd.DataFrame:
    """
    Samples n time series per class label based on their unique_id and returns the corresponding rows.

    Parameters:
    - df: pd.DataFrame with columns ['unique_id', 'ds', 'y', 'label']
    - n_samples: Number of time series to sample per class (label).

    Returns:
    - pd.DataFrame: A DataFrame with sampled time series.
    """
    sampled_ids = df.groupby("label")["unique_id"].apply(
        lambda x: x.drop_duplicates().sample(n=n_samples, random_state=random_state)
    )
    df_sampled = df[df["unique_id"].isin(sampled_ids)]
    return df_sampled


def vis_time_series_classification_samples(
    df: pd.DataFrame,
    n_samples_per_label: int = 5,
    alpha: float = 0.8,
    n_col: int = 1,
    random_state: int = None,
) -> ggplot:
    """
    Visualize a sample of n time series per class label from the given DataFrame.

    Parameters:
    - df: pd.DataFrame with columns ['unique_id', 'ds', 'y', 'label']
    - n_samples_per_label: int, number of samples to visualize
    """
    df_sampled = sample_ts_per_class(df, n_samples_per_label, random_state)

    # add color_id col to visualise samples per class label
    df_sampled = df_sampled.assign(
        color_id=df_sampled.groupby("label")["unique_id"]
        .transform(lambda x: pd.factorize(x)[0] + 1)
        .astype(str)
    )
    df_sampled["label_text"] = df_sampled["label"].apply(lambda x: f"Label: {x}")

    tooltip = (
        layer_tooltips()
        .line("ds| @ds")
        .line("y| @y")
        .line("unique_id| @unique_id")
        .format("y", " .2f")
    )

    colors = ScientificPalette.get_palette(df_sampled["color_id"].nunique())

    # Plot the time series data
    p = (
        ggplot(df_sampled, aes("ds", "y", group="unique_id", color="color_id"))
        + geom_line(size=1, alpha=alpha, tooltips=tooltip)
        + facet_wrap("label_text", ncol=n_col, scales="free_y")
        + scale_color_manual(values=colors)
        + labs(
            x="ds",
            y="y",
            title=f"Sampled Time Series Per Class (N={n_samples_per_label})",
        )
        + theme_light()
        + theme(legend_position="none")
        + labs(x="Time Stamp", y="Value")
    )

    return p


def plot_confusion_matrix(
    conf_matrix: np.ndarray, class_names: Optional[list] = None
) -> ggplot:
    """Plots confusion matrix using lets_plot.

    Args:
        conf_matrix: Confusion matrix as numpy array
        class_names: Optional list of class names. If None, will use indices
    """
    if class_names is None:
        class_names = [str(i) for i in range(conf_matrix.shape[0])]

    # Create dataframe for plotting
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=[f"Actual {name}" for name in class_names],
        columns=[f"Predicted {name}" for name in class_names],
    )
    conf_matrix_melted = conf_matrix_df.reset_index().melt(id_vars="index")
    conf_matrix_melted.columns = ["Actual", "Predicted", "Count"]

    colors = ScientificPalette.get_palette(1)
    p = (
        ggplot(conf_matrix_melted)
        + geom_tile(aes("Predicted", "Actual", fill="Count"), alpha=0.8)
        + geom_text(aes("Predicted", "Actual", label="Count"), color="black")
        + scale_fill_gradient(low="white", high=colors[0])
        + theme_minimal()
        + theme(axis_title=element_blank())
    )
    return p


def plot_ts_attributions(
    ts_df: pd.DataFrame,
    x_col: str = "ds",
    y_col: str = "y",
    exp: Iterable[float] | str = "r",
    facet_col: str = None,
    n_col: int = 1,
    title: str = None,
    subtitle: str = None,
    exp_label: str = "Relevance (r)",
    hm_theme: str = "inferno",
    hm_alpha: float = 1,
    ts_color: str = "white",
    ts_size: int = 1,
    plot_size: tuple[int] = (1500, 525),
    x_lab: str = None,
    y_lab: str = None,
    y_lim: tuple[int] = None,
    scale_fill_limits: tuple[float] = [0, 1],
    tooltip: bool = True,
):
    """
    Plot time series data with relevance scores as a heatmap overlay.

    Parameters:
    - ts_df (pd.DataFrame): The time series data to plot.
    - exp (Iterable[float]): The relevance scores corresponding to each
        data point. If None, only the time series line plot will be shown.
    - facet_col (str): The column name for faceting the plot.
    - n_col (int): The number of columns for faceting. Default is 1.
    - title (str): The title of the plot.
    - subtitle (str): The subtitle of the plot.
    - x_col (str): The column name for the x-axis values in ts_df.
    - y_col (str): The column name for the y-axis values in ts_df.
    - exp_label (str): The label for the relevance scores in the heatmap.
    - hm_theme (str): The color theme for the heatmap. Default is "inferno".
    - hm_alpha (float): The transparency level for the heatmap. Default is 1.
    - ts_color (str): The color of the time series line plot. Default is "grey".
    - ts_size (int): The size of the time series line plot. Default is 1.
    - plot_size (tuple[int]): The size of the plot in pixels. Default is (1000, 400).
    - x_lab (str): The label for the x-axis.
    - y_lab (str): The label for the y-axis.
    - y_lim (tuple[int]): The limits for the y-axis. Default is None.
    - scale_fill_limits (tuple[float]): The limits for the color scale. Default is [0, 1].
    - tooltip (bool): Whether to show tooltips on hover. Default is True.

    Returns:
    - p (ggplot): The plot object containing the time series plot with relevance heatmap overlay.
    """
    ts_df = ts_df.copy()

    # make scale fill limits inclusive
    scale_fill_limits = [scale_fill_limits[0] - 1e-3, scale_fill_limits[1] + 1e-3]

    default_tooltip = (
        layer_tooltips()
        .line(f"x| @{x_col}")
        .line(f"y| @{y_col}")
        .line(f"r| @{exp}")
        .format(exp, " .3f")
        .format(x_col, " .0f")
        .format(y_col, " .3f")
    )

    # set up plot object
    p = (
        ggplot()
        + theme_minimal2()
        + ggsize(plot_size[0], plot_size[1])
        + theme(
            legend_position="bottom",
            axis_title="blank" if not x_lab and not y_lab else None,
        )
    )
    p = p + ggtitle(title, subtitle) if title else p
    p = p + xlab(x_lab) if x_lab else p
    p = p + ylab(y_lab) if y_lab else p
    p = p + facet_wrap(facet_col, ncol=n_col) if facet_col else p

    # plot heatmap over line plot if there are relevance scores
    if exp is not None:
        # Calculate the boundaries for the rectangles
        ts_df["xmin"] = ts_df[x_col] - 0.5
        ts_df["xmax"] = ts_df[x_col] + 0.5
        ts_df["ymin"] = ts_df[y_col].min() if not y_lim else y_lim[0]
        ts_df["ymax"] = ts_df[y_col].max() if not y_lim else y_lim[1]
        ts_df[exp_label] = ts_df[exp] if isinstance(exp, str) else exp

        # Calculate x-axis limits
        x_min = ts_df[x_col].min() - 0.5
        x_max = ts_df[x_col].max() + 0.5

        # add heatmap to plot
        p += (
            geom_rect(
                aes(
                    xmin="xmin",
                    xmax="xmax",
                    ymin="ymin",
                    ymax="ymax",
                    fill=exp_label,
                ),
                data=ts_df,
                alpha=hm_alpha,
                size=0,
                tooltips=default_tooltip if tooltip else None,
            )
            + scale_fill_viridis(
                name=exp_label, option=hm_theme, limits=scale_fill_limits
            )
            + coord_cartesian(xlim=[x_min, x_max])
        )

    # plot time series
    p += geom_line(
        aes(x=x_col, y=y_col),
        data=ts_df,
        color=ts_color,
        size=ts_size,
        tooltips=default_tooltip if tooltip else None,
    )

    # set y-axis limits if provided
    p = p + ylim(y_lim[0], y_lim[1]) if y_lim else p

    return p


def plot_perturbation_analysis(
    explainer: "TimeSeriesExplainer",
    sample_idx: int,
    method: str,
    label: Union[str, int] = "predicted",
    perturbation_strategy: str = "zero",
    plot_between: bool = True,
    rescale: bool = False,
    print_scores: bool = True,
    show_plot: bool = True,
    auc_types: Optional[list] = None,
    perturbation_ratio: float = 0.5,
    features_per_step: int = 1,
    k: float = 0.1,
) -> None:
    """Convenience function to show perturbation evaluation results for a given observation.

    Handles visualization and metric computation for perturbation analysis, adapting
    the output based on available metrics and curves.
    Visualises the perturbation curves

    Args:
        explainer: TimeSeriesExplainer instance
        sample_idx: Index of sample to explain
        method: Attribution method to use
        label: Label to explain - can be "predicted", "true", or specific class index
        perturbation_strategy: Strategy for perturbing features (e.g., "zero", "mean")
        plot_between: If True, shows area between MoRF and LeRF curves in gray.
                     Falls back to individual areas if only one curve available.
        rescale: Whether to rescale attributions
        print_scores: Whether to print evaluation metrics
        show_plot: Whether to show the visualization
        auc_types: List of AUC types to compute. Defaults to ["MoRF", "LeRF"]
                  Valid options are "MoRF", "LeRF" or both.
        perturbation_ratio: Maximum perturbation coverage.
        features_per_step: Number of features to perturb at each step.
        k: Fraction of top-k features to perturb

    Notes:
        - DS and DDS scores are only computed if both MoRF and LeRF curves are available
        - Warnings are printed if requested metrics cannot be computed
        - The plot layout adapts based on available metrics and curves
    """
    # Input validation
    if auc_types is None:
        auc_types = ["MoRF", "LeRF"]

    valid_auc_types = {"MoRF", "LeRF"}
    if not all(auc_type in valid_auc_types for auc_type in auc_types):
        raise ValueError(f"Invalid AUC type. Must be one of {valid_auc_types}")

    def _get_target_label(explanation_result):
        """Determine which label to evaluate based on input parameters."""
        if label == "predicted":
            return explanation_result.prediction.label
        elif label == "true":
            return explanation_result.metadata.true_label
        return label

    def _create_metrics_table(result):
        """Create metrics table based on available scores."""
        headers = []
        values = []

        # Add available AUC scores
        for auc_type in auc_types:
            if auc_type in result.auc_scores:
                headers.append(f"AUC {auc_type}")
                values.append(f"{result.auc_scores[auc_type]:.3f}")

        # Add DS and DDS if both curves available
        both_curves = "MoRF" in result.auc_scores and "LeRF" in result.auc_scores
        if both_curves:
            try:
                ds_score = result.calculate_ds()
                headers.append("DS Score")
                values.append(f"{ds_score:.3f}")

            except Exception as e:
                print(f"Warning: Error calculating DS/DDS scores: {str(e)}")

        return [headers, values]

    try:
        # Generate explanation
        explanation = explainer.explain(
            sample_index=sample_idx, method=method, label=label, rescale=rescale
        )

        # Get target label for evaluation
        label_to_eval = _get_target_label(explanation)

        # Initialize evaluator and run evaluation
        evaluator = PerturbationEvaluator(
            features_per_step=features_per_step,
            auc_types=auc_types,
            perturbation_ratio=perturbation_ratio,
        )
        result = evaluator.evaluate(
            explainer=explainer,
            sample_index=sample_idx,
            attributions=explanation.data.attributions,
            target_class=label_to_eval,
            perturbation_strategy=perturbation_strategy,
            k=k,
        )

        # Show results if requested
        if print_scores:
            metrics = _create_metrics_table(result)
            if metrics[0]:  # Only print if we have metrics
                print(
                    tabulate.tabulate(
                        metrics, headers="firstrow", tablefmt="fancy_grid"
                    )
                )

        # Create and show plots if requested
        if show_plot:
            plots = []

            # Attribution heatmap
            heatmap_plot = explanation.visualize_attributions()
            plots.append(heatmap_plot)

            # Perturbation analysis curves
            if result.scores:  # Only create plot if we have scores
                perturbation_plot = plot_perturbation_analysis_curves(
                    result,
                    perturbation_ratio=perturbation_ratio,
                    title=f"Perturbation Analysis (strategy: {perturbation_strategy})",
                    plot_between=plot_between,
                    include_ds=len(auc_types) > 1,
                    include_dds=len(auc_types) > 1,
                )
                plots.append(perturbation_plot)

            # Combine and display plots
            if plots:
                combined_plot = gggrid(plots, ncol=1) + ggsize(800, 700)
                combined_plot.show()

    except Exception as e:
        print(f"Error generating functional evaluation results: {str(e)}")
        raise


def plot_metric_distributions(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_levels: List[str],
    color_by: Optional[str] = None,
    y_limits: Tuple[float, float] = None,
    x_label: str = None,
    y_label: str = None,
    title: Optional[str] = None,
    plot_size: Tuple[int, int] = None,
) -> ggplot:
    """Create violin plot showing distribution of degradation scores across attribution methods.

    Args:
        data (pd.DataFrame): DataFrame containing experiment results
        attribution_methods (List[str]): List of attribution methods to include
        x_col (str): Column name for x-axis (e.g., "attribution_method")
        y_col (str): Column name for y-axis (e.g., "degradation_score")
        x_levels (List[str]): List of levels for x-axis. Used for ordering x-axis.
        color_by (Optional[str]): Column name for coloring (e.g., "true_label")
        y_limits (Tuple[float, float]): Y-axis limits
        x_label (str): X-axis label
        y_label (str): Y-axis label
        title (Optional[str]): Plot title
        plot_size (Tuple[int, int]): Plot size in pixels

    Returns:
        ggplot: Distribution plot
    """
    plot_data = data.copy()
    attribution_var = as_discrete(x_col, order=1, levels=x_levels)

    base_plot = (
        ggplot(plot_data, aes(x=attribution_var, y=y_col))
        + theme_minimal2()
        + geom_hline(yintercept=0, linetype="dashed", alpha=0.7)
        + theme(
            legend_position=[0, 1],
            legend_justification=[0, 1],
            legend_direction="horizontal",
            legend_background=element_rect(color="grey", size=0.1),
        )
    )

    if y_limits:
        base_plot += scale_y_continuous(limits=y_limits)

    if color_by:
        plot_data[color_by] = plot_data[color_by].astype(str)
        plot_label = color_by.replace("_", " ").title()
        n_colors = plot_data[color_by].nunique()
        plot = base_plot + (
            geom_violin(
                aes(fill=as_discrete(color_by, order=1)),
                alpha=0.5,
                width=0.8,
                scale="width",
            )
            + geom_boxplot(
                aes(fill=as_discrete(color_by, order=1)),
                width=0.15,
                color="black",
                alpha=1,
                outlier_size=0,
                scale="width",
            )
            + labs(y=y_label, title=title, fill=plot_label, x=x_label)
            + scale_fill_manual(values=ScientificPalette.get_palette(n_colors))
        )
    else:
        fill_color = ScientificPalette.get_palette(1)[0]
        plot = base_plot + (
            geom_violin(alpha=0.5, width=0.8, scale="width", fill=fill_color)
            + geom_boxplot(
                width=0.1,
                color="black",
                alpha=1,
                outlier_size=0,
                scale="width",
                fill=fill_color,
            )
            + labs(x=x_label, y=y_label, title=title)
        )

    if plot_size:
        plot = plot + ggsize(plot_size[0], plot_size[1])

    return plot


def plot_parallel_coords(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    instance_id_col: Optional[str] = None,
    order: Literal["ascending", "descending", "name"] = "descending",
    line_alpha: float = 0.07,
    point_size: float = 2,
    show_means: bool = True,
    mean_line_size: float = 3,
    mean_alpha: float = 1,
    show_individual: bool = True,
    x_label: str = None,
    y_label: str = None,
    title: Optional[str] = None,
    plot_size: Tuple[int, int] = None,
) -> ggplot:
    """Create a parallel coordinates plot with ordered categories.

    Args:
        data (pd.DataFrame): Input DataFrame containing the data to plot.
        x_col (str): Column name for x-axis (categories). Defaults to "perturbation_strategy".
        y_col (str): Column name for y-axis values. Defaults to "ds".
        group_col (str): Column name for grouping/coloring. Defaults to "true_label".
        instance_id_col (Optional[str]): Column name for instance IDs. If None, will be created.
        order (Literal["ascending", "descending", "name"]): Order of x-axis categories.
            - "ascending": Sort by group differences in ascending order
            - "descending": Sort by group differences in descending order
            - "name": Sort by category names
        line_alpha (float): Alpha value for lines. Defaults to 0.05.
        point_size (float): Size of points. Defaults to 2.
        show_means (bool): Whether to show mean lines. Defaults to True.
        mean_line_size (float): Size of mean lines. Defaults to 2.5.
        mean_alpha (float): Alpha value for mean lines. Defaults to 1.
        show_individual (bool): Whether to show individual points and lines. Defaults to True.
        x_label (str): X-axis label. Defaults to None.
        y_label (str): Y-axis label. Defaults to None.
        title (Optional[str]): Plot title. Defaults to None.
        plot_size (Tuple[int, int]): Plot size in pixels. Defaults to None.

    Returns:
        ggplot: The resulting plot object.

    Raises:
        ValueError: If order is not one of "ascending", "descending", or "name".
    """
    if order not in ["ascending", "descending", "name"]:
        raise ValueError("order must be one of 'ascending', 'descending', or 'name'")

    plot_data = data.copy()
    if instance_id_col is None:
        plot_data["instance_id"] = plot_data.groupby([group_col]).cumcount()
        instance_id_col = "instance_id"

    def natural_sort_key(x):
        """Convert string to float if possible, otherwise return string.

        This ensures proper sorting of numeric strings including negatives and decimals.
        """
        try:
            return float(x)  # Convert to float to handle both integers and decimals
        except (ValueError, TypeError):
            return x

    # Determine category ordering
    if order == "name":
        ordered_categories = sorted(plot_data[x_col].unique(), key=natural_sort_key)
    else:
        means = plot_data.groupby([x_col, group_col])[y_col].mean().unstack()
        if means.shape[1] == 2:
            diffs = abs(means.iloc[:, 0] - means.iloc[:, 1])
        else:
            diffs = means.apply(lambda row: max(row) - min(row), axis=1)
        ordered_categories = diffs.sort_values(
            ascending=(order == "ascending")
        ).index.tolist()

    # Calculate means if needed
    if show_means:
        mean_data = plot_data.groupby([x_col, group_col])[y_col].mean().reset_index()
        individual_alpha = min(line_alpha, 0.2) if show_individual else 0
    else:
        individual_alpha = line_alpha if show_individual else 0

    # Create base plot
    n_color_groups = data[group_col].nunique()
    colors = ScientificPalette.get_palette(n_color_groups)
    color_levels = sorted(data[group_col].unique())
    plot = (
        ggplot(
            plot_data,
            aes(
                x=as_discrete(x_col),
                y=y_col,
                color=as_discrete(group_col, levels=color_levels),
                group=instance_id_col,
            ),
        )
        + scale_x_discrete(limits=ordered_categories)
        + labs(
            title=title, x=x_label, y=y_label, color=group_col.replace("_", " ").title()
        )
        + theme_minimal2()
        + theme(
            legend_position=[0, 1],
            legend_justification=[0, 1],
            legend_direction="horizontal",
            legend_background=element_rect(color="grey", size=0.1),
        )
        + scale_color_manual(values=colors)
    )

    # Add individual points and lines if requested
    if show_individual:
        plot = plot + geom_line(alpha=individual_alpha)
        plot = plot + geom_point(size=point_size, alpha=individual_alpha)

    # Add mean lines if requested
    if show_means:
        plot = (
            plot
            + geom_line(
                data=mean_data,
                mapping=aes(
                    x=as_discrete(x_col),
                    y=y_col,
                    color=as_discrete(group_col),
                    group=group_col,
                ),
                size=mean_line_size,
                alpha=mean_alpha,
            )
            + geom_point(
                data=mean_data,
                mapping=aes(
                    x=as_discrete(x_col),
                    y=y_col,
                    color=as_discrete(group_col),
                    group=group_col,
                ),
                size=mean_line_size * 2.5,
                alpha=mean_alpha,
            )
        )

    # Add zero line
    plot = plot + geom_hline(yintercept=0, linetype="dashed", alpha=0.7, color="black")

    if plot_size:
        plot = plot + ggsize(plot_size[0], plot_size[1])

    return plot
