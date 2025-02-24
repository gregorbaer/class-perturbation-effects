import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from tsxai.attribution.evaluation.metrics import calculate_class_adjusted_metric
from tsxai.utils.logging import setup_logger
from tsxai.utils.results import load_and_format_experiment_results

# Setup logging
logger = setup_logger(
    __name__, logging.DEBUG, enable_file_logging=False, log_dir="logs"
)

# Configuration constants
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PERTURBATIONS_DIR = PROJECT_ROOT / "results/perturbation_results"
RESULTS_DIR = PERTURBATIONS_DIR / "paper_results/tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = PERTURBATIONS_DIR / "paper_results/compiled_results.csv"

# Configuration dictionaries for sorting orders
MODEL_ORDER = ["ResNet", "InceptionTime"]
DATASET_ORDER = ["FordA", "FordB", "Wafer", "ElectricDevices"]
ATTRIBUTION_ORDER = ["GR", "IG", "SG", "GS", "FO"]

# Column mappings for consistency
COLUMN_MAPPINGS = {
    "dataset_name": "Dataset",
    "model_name": "Model",
    "attribution_method": "Attribution",
    "perturbation_strategy": "Perturbation",
}

# Constant perturbation values in correct order
CONSTANT_PERTURBATION_VALUES = [
    "-2",
    "-1.5",
    "-1",
    "-0.5",
    "0",
    "0.5",
    "1",
    "1.5",
    "2",
]


def sort_multiindex_with_numeric_perturbations(table: pd.DataFrame) -> pd.DataFrame:
    """Sort a table with MultiIndex where second level contains numeric values.

    Args:
        table (pd.DataFrame): Table with MultiIndex (Dataset, Perturbation)

    Returns:
        pd.DataFrame: Sorted table
    """
    # Create a sorting key function for the row multi-index
    dataset_sort_map = {ds: idx for idx, ds in enumerate(DATASET_ORDER)}

    def get_row_sort_key(idx_tuple):
        dataset, perturbation = idx_tuple
        # Get the position in the dataset ordering
        dataset_pos = dataset_sort_map.get(dataset, len(DATASET_ORDER))
        # Convert perturbation to float for numeric sorting
        try:
            perturbation_val = float(perturbation)
        except ValueError:
            perturbation_val = float("inf")  # Non-numeric values go to the end
        return (dataset_pos, perturbation_val)

    return table.reindex(sorted(table.index, key=get_row_sort_key))


def format_metric_table(
    df: pd.DataFrame,
    metric_col: str,
    index_cols: List[str],
    pivot_cols: List[str],
    column_order: Optional[List[Tuple]] = None,
    dataset_order: Optional[List[str]] = None,
    numeric_perturbations: bool = False,
) -> pd.DataFrame:
    """Format metrics DataFrame into a pivot table with custom ordering.

    Args:
        df (pd.DataFrame): DataFrame with calculated metrics
        metric_col (str): Name of the metric column to pivot
        index_cols (List[str]): Columns to use as index
        pivot_cols (List[str]): Columns to use as columns
        column_order (Optional[List[Tuple]]): Order of columns in pivot table
        dataset_order (Optional[List[str]]): Order of datasets
        numeric_perturbations (bool): Whether to sort perturbations numerically

    Returns:
        pd.DataFrame: Formatted pivot table
    """
    # Create pivot table
    formatted_table = pd.pivot_table(
        df,
        values=metric_col,
        index=index_cols,
        columns=pivot_cols,
        aggfunc="mean",
    )

    # Create and filter column order
    if column_order is not None:
        valid_cols = [col for col in column_order if col in formatted_table.columns]
        formatted_table = formatted_table.reindex(columns=valid_cols)

    # Sort rows
    if dataset_order is not None:
        dataset_sort_map = {ds: idx for idx, ds in enumerate(dataset_order)}

        def get_row_sort_key(idx_tuple):
            dataset, perturbation = idx_tuple
            dataset_pos = dataset_sort_map.get(dataset, len(dataset_order))
            if numeric_perturbations:
                try:
                    perturbation_val = float(perturbation)
                except ValueError:
                    perturbation_val = float("inf")
                return (dataset_pos, perturbation_val)
            return (dataset_pos, perturbation)

        formatted_table = formatted_table.reindex(
            sorted(formatted_table.index, key=get_row_sort_key)
        )

    return formatted_table


def main():
    """Main function to process results and generate CSV tables."""
    # Setup logging
    logger = setup_logger(
        __name__, logging.DEBUG, enable_file_logging=False, log_dir="logs"
    )

    # Common parameters
    metric_kwargs = {
        "metric_col": "ds",
        "group_cols": ["Dataset", "Model", "Attribution", "Perturbation"],
        "class_col": "true_label",
    }

    format_kwargs = {
        "metric_col": "adjusted_metric",
        "index_cols": ["Dataset", "Perturbation"],
        "pivot_cols": ["Model", "Attribution"],
        "column_order": [(m, a) for m in MODEL_ORDER for a in ATTRIBUTION_ORDER],
        "dataset_order": DATASET_ORDER,
    }

    # Alpha values for adjusted and unadjusted metrics
    alpha_adjusted = 1
    alpha_unadjusted = 0
    logger.info(f"Using alpha={alpha_adjusted} for class-adjusted DS.")

    # Process named perturbation strategies
    logger.info("Processing named perturbation strategies...")
    named_results_df = load_and_format_experiment_results(RESULTS_PATH, "named")
    named_results_df = named_results_df.rename(columns=COLUMN_MAPPINGS)

    # Log dataset information for named perturbations
    logger.info("Named perturbation statistics:")
    logger.info(f"Models: {named_results_df['Model'].unique()}")
    logger.info(f"Attributions: {named_results_df['Attribution'].unique()}")
    logger.info(f"Datasets: {named_results_df['Dataset'].unique()}")
    logger.info(f"Perturbations: {named_results_df['Perturbation'].unique()}")
    logger.info(f"Dimensions: {named_results_df.shape}")

    # Generate and save named perturbation tables
    logger.info("Generating average DS table for named perturbations...")
    # Calculate unadjusted metrics (alpha=0)
    named_average_ds = calculate_class_adjusted_metric(
        df=named_results_df, alpha=alpha_unadjusted, **metric_kwargs
    )
    # Format table
    named_average_ds_table = format_metric_table(df=named_average_ds, **format_kwargs)
    named_average_ds_table.to_csv(RESULTS_DIR / "average_ds_table.csv")
    logger.info("Average DS table saved.")

    logger.info("Generating class-adjusted DS table for named perturbations...")
    # Calculate class-adjusted metrics
    named_adjusted_ds = calculate_class_adjusted_metric(
        df=named_results_df, alpha=alpha_adjusted, **metric_kwargs
    )
    # Format table
    named_adjusted_ds_table = format_metric_table(df=named_adjusted_ds, **format_kwargs)
    named_adjusted_ds_table.to_csv(RESULTS_DIR / "adjusted_ds_table.csv")
    logger.info("Adjusted DS table saved.")

    # Process constant perturbation strategies
    logger.info("Processing constant perturbation strategies...")
    constant_results_df = load_and_format_experiment_results(RESULTS_PATH, "constant")
    constant_results_df = constant_results_df.rename(columns=COLUMN_MAPPINGS)

    # Log dataset information for constant perturbations
    logger.info("Constant perturbation statistics:")
    logger.info(f"Models: {constant_results_df['Model'].unique()}")
    logger.info(f"Attributions: {constant_results_df['Attribution'].unique()}")
    logger.info(f"Datasets: {constant_results_df['Dataset'].unique()}")
    logger.info(f"Perturbations: {constant_results_df['Perturbation'].unique()}")
    logger.info(f"Dimensions: {constant_results_df.shape}")

    # Update format kwargs for constant perturbations
    constant_format_kwargs = {**format_kwargs, "numeric_perturbations": True}

    # Generate and save constant perturbation tables
    logger.info("Generating average DS table for constant perturbations...")
    # Calculate unadjusted metrics (alpha=0)
    constant_average_ds = calculate_class_adjusted_metric(
        df=constant_results_df, alpha=alpha_unadjusted, **metric_kwargs
    )
    # Format table with numeric sorting for perturbations
    constant_average_ds_table = format_metric_table(
        df=constant_average_ds, **constant_format_kwargs
    )
    constant_average_ds_table.to_csv(RESULTS_DIR / "constant_average_ds_table.csv")
    logger.info("Average DS table saved.")

    logger.info("Generating adjusted DS table for constant perturbations...")
    # Calculate class-adjusted metrics
    constant_adjusted_ds = calculate_class_adjusted_metric(
        df=constant_results_df, alpha=alpha_adjusted, **metric_kwargs
    )
    # Format table with numeric sorting for perturbations
    constant_adjusted_ds_table = format_metric_table(
        df=constant_adjusted_ds, **constant_format_kwargs
    )
    constant_adjusted_ds_table.to_csv(RESULTS_DIR / "constant_adjusted_ds_table.csv")
    logger.info("Adjusted DS table saved.")


if __name__ == "__main__":
    main()
