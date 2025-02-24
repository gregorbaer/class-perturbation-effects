import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import pandas as pd


class ResultsManager:
    """Manages saving and loading of benchmark results.

    Args:
        base_dir (Union[str, Path]): Base directory for saving results.
        experiment_name (str): Name of the experiment for organizing results.
        logger (logging.Logger): Logger instance for tracking operations.
        cleanup_intermediate (bool, optional): Whether to delete intermediate results
            after compilation. Defaults to True.

    Attributes:
        results_dir (Path): Directory for storing all results.
        individual_results_dir (Path): Directory for individual experiment results.
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        experiment_name: str,
        logger: logging.Logger,
        cleanup_intermediate: bool = True,
    ):
        self.base_dir = Path(base_dir)
        self.logger = logger
        self.cleanup_intermediate = cleanup_intermediate

        # Create timestamp-based experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.base_dir / f"{experiment_name}_{timestamp}"
        self.individual_results_dir = self.results_dir / "individual_results"

        # Create directory structure
        self.individual_results_dir.mkdir(parents=True, exist_ok=True)

    def save_individual_result(
        self,
        result: pd.DataFrame,
        dataset_name: str,
        model_name: str,
        attribution_method: str,
        perturbation_strategy: str,
    ) -> None:
        """Save individual experiment results."""
        fname = f"{dataset_name}_{model_name}_{attribution_method}_{perturbation_strategy}.csv"
        save_path = self.individual_results_dir / fname
        result.to_csv(save_path, index=False)
        self.logger.debug(f"Saved individual result to {save_path}")

    def compile_and_save_results(self) -> None:
        """Compile all individual results into a single DataFrame and save to csv.

        If cleanup_intermediate is True, deletes the individual_results_dir after
        successful compilation.
        """
        all_results = []
        for result_file in self.individual_results_dir.glob("*.csv"):
            result = pd.read_csv(result_file)
            all_results.append(result)

        compiled_results = pd.concat(all_results, ignore_index=True)
        compiled_path = self.results_dir / "compiled_results.csv"

        compiled_results.to_csv(compiled_path, index=False)
        self.logger.info(f"Saved compiled results to {compiled_path}")

        if self.cleanup_intermediate:
            try:
                shutil.rmtree(self.individual_results_dir)
                self.logger.info("Cleaned up intermediate results directory")
            except Exception as e:
                self.logger.warning(f"Failed to clean up intermediate results: {e}")


def load_and_format_experiment_results(
    results_path: str,
    perturbation_mode: Literal["named", "constant"] = "named",
    attribution_mappings: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Load and preprocess experimental results from XAI perturbation analysis.

    Helps to format experimental result into consistent format used in paper.

    Args:
        results_path (str): Path to the CSV file containing experimental results.
        perturbation_mode (str): Either "named" for named perturbations (like gaussian_noise)
            or "constant" for fixed-value perturbations.
        attribution_mappings (Optional[Dict[str, str]], optional): Dictionary mapping
            attribution method names. Defaults to predefined mappings.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with mapped names and filtered perturbations.
    """
    DEFAULT_ATTRIBUTION_MAPPINGS = {
        "GRAD": "GR",
        "IG": "IG",
        "SG": "SG",
        "GS": "GS",
        "FO": "FO",
    }

    NAMED_PERTURBATION_MAPPINGS = {
        "gaussian_noise": "Gauss",
        "uniform_noise": "Unif",
        "subsequence_mean": "SubMean",
        "inverse": "Inv",
        "opposite": "Opp",
        "0": "Zero",
    }

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
    CONSTANT_PERTURBATION_MAPPINGS = {val: val for val in CONSTANT_PERTURBATION_VALUES}

    perturbation_mappings = (
        NAMED_PERTURBATION_MAPPINGS
        if perturbation_mode == "named"
        else CONSTANT_PERTURBATION_MAPPINGS
    )

    attribution_mappings = attribution_mappings or DEFAULT_ATTRIBUTION_MAPPINGS

    results = pd.read_csv(results_path)

    if any(
        col not in results.columns
        for col in ["attribution_method", "perturbation_strategy"]
    ):
        raise ValueError("Missing required columns")

    # Modified approach - determine type based on inclusion in mappings
    if perturbation_mode == "named":
        # Keep only rows with perturbation strategies in NAMED_PERTURBATION_MAPPINGS
        # (including "0" which is mapped to "Zero")
        results = results[
            results["perturbation_strategy"]
            .astype(str)
            .isin(list(NAMED_PERTURBATION_MAPPINGS.keys()))
        ]
    else:  # constant mode
        # Keep only rows with perturbation strategies that match numeric pattern
        results = results[
            results["perturbation_strategy"]
            .astype(str)
            .apply(lambda x: bool(re.match(r"^-?\d*\.?\d+$", str(x))))
        ]

    results["attribution_method"] = results["attribution_method"].replace(
        attribution_mappings
    )
    results["perturbation_strategy"] = results["perturbation_strategy"].replace(
        perturbation_mappings
    )

    return results
