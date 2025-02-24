from pathlib import Path

import pandas as pd
import torch

from tsxai.attribution.evaluation.metrics import evaluate_correctness_for_samples
from tsxai.attribution.explainer import TimeSeriesExplainer
from tsxai.data.loading import load_ucr_data
from tsxai.data.utils import sample_equal_per_class
from tsxai.modelling.utils import load_trained_model
from tsxai.utils.config import load_config
from tsxai.utils.logging import setup_logger
from tsxai.utils.results import ResultsManager

# Get project root directory (2 levels up from script)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def main():
    # load config
    config = load_config(PROJECT_ROOT / "scripts/config.yaml")
    MODELS_DIR = PROJECT_ROOT / config["paths"]["models_dir"]
    PERTURBATIONS_DIR = PROJECT_ROOT / config["paths"]["perturbations_dir"]
    DATASETS = config["datasets"]

    train_config = config["train_models"]
    train_config["model_config"]["model_dir"] = str(MODELS_DIR)
    pconf = config["perturbation_experiments"]

    perturbation_strategies = (
        pconf["perturbation_strategies"] + pconf["perturbation_constants"]
    )

    # create results dir if it does not exist
    PERTURBATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # setup logger
    logger = setup_logger(
        __name__,
        enable_file_logging=pconf["save_logs"],
        log_dir=PERTURBATIONS_DIR,
        log_file="perturbation_experiments.log",
    )

    # Initialize results manager
    results_manager = ResultsManager(
        base_dir=PERTURBATIONS_DIR,
        experiment_name="perturbation_benchmark",
        logger=logger,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting experiments with device: {device}")

    # check for each dataset if trained models are available
    modelweight_paths = list(MODELS_DIR.rglob("*.ckpt"))
    modelweight_paths = [
        f
        for f in modelweight_paths
        for model_name in config["models"]
        for dataset_name in DATASETS
        if model_name in str(f) and dataset_name in str(f)
    ]
    if not len(modelweight_paths) == len(DATASETS) * len(config["models"]):
        msg = "Not all datasets have trained models or there are multiple matching models."
        logger.error(msg)
        raise ValueError(msg)

    # start benchmark
    for i_dataset, dataset_name in enumerate(DATASETS, 1):
        # load data
        which_dataset = f"{dataset_name} [{i_dataset}/{len(DATASETS)}]"
        data = load_ucr_data(dataset_name, remap_labels_ascending=True)
        config["dataset_name"] = dataset_name

        # sample observations from test data to evaluate explanations from
        sample_ids = sample_equal_per_class(
            data.X_test, data.y_test, pconf["n_attributions_per_class"], pconf["seed"]
        )

        # calculate number of features to perturb per step
        features_per_step = max(
            1, round(data.n_timesteps * pconf["features_per_step_ratio"])
        )

        # load trained prediction models
        models_dict = {
            "ResNet": load_trained_model(data, dataset_name, "ResNet", MODELS_DIR),
            "InceptionTime": load_trained_model(
                data, dataset_name, "InceptionTime", MODELS_DIR
            ),
        }

        for model_name in config["models"]:
            logger.info(
                f"Running perturbation analysis for dataset {which_dataset} on {model_name}. "
                f"(Features per step: {features_per_step}, perturbation ratio: {pconf['max_perturbation_ratio']})"
            )
            explainer = TimeSeriesExplainer(dataset=data, model=models_dict[model_name])

            for i_perturb, perturbation_strategy in enumerate(
                perturbation_strategies, 1
            ):
                for i_attr, attribution_method in enumerate(
                    pconf["attribution_methods"], 1
                ):
                    logger.info(
                        f"Evaluating [{i_attr}/{len(pconf['attribution_methods'])}]:"
                        f"[{attribution_method}] attributions for {model_name} "
                        f"for dataset={which_dataset} "
                        f"(perturbation={perturbation_strategy} "
                        f"[{i_perturb}/{len(perturbation_strategies)}])"
                    )

                    # Run evaluation
                    funceval_results: pd.DataFrame = evaluate_correctness_for_samples(
                        explainer=explainer,
                        sample_ids=sample_ids,
                        method=attribution_method,
                        label=pconf["label_to_explain"],
                        perturbation_strategy=perturbation_strategy,
                        features_per_step=features_per_step,
                        perturbation_ratio=pconf["max_perturbation_ratio"],
                    )

                    # Add metadata
                    funceval_results = funceval_results.assign(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        attribution_method=attribution_method,
                        perturbation_strategy=perturbation_strategy,
                        n_attributions=pconf["n_attributions_per_class"],
                    )

                    # Save results
                    results_manager.save_individual_result(
                        result=funceval_results,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        attribution_method=attribution_method,
                        perturbation_strategy=perturbation_strategy,
                    )

    # Compile all results at the end
    results_manager.compile_and_save_results()
    logger.info("Benchmark finished successfully.")


if __name__ == "__main__":
    main()
