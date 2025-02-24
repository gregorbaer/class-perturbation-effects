from pathlib import Path

import pandas as pd
import torch

from tsxai.data.dataloaders import prepare_dataloaders
from tsxai.data.loading import load_ucr_data
from tsxai.modelling.evaluation import evaluate_model
from tsxai.modelling.models import InceptionTime, ResNet
from tsxai.modelling.training import setup_deterministic_training, train_model
from tsxai.utils.config import load_config
from tsxai.utils.logging import setup_logger

# Get project root directory (2 levels up from script)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def main():
    # load config
    config = load_config(PROJECT_ROOT / "scripts/config.yaml")
    MODELS_DIR = PROJECT_ROOT / config["paths"]["models_dir"]
    train_config = config["train_models"]
    train_config["model_config"]["model_dir"] = str(MODELS_DIR)

    # setup logger
    logger = setup_logger(
        __name__,
        enable_file_logging=train_config["save_logs"],
        log_dir=MODELS_DIR,
        log_file="train_models.log",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training pipeline with device: {device}")
    logger.info(f"Model configuration: {train_config['model_config']}")
    logger.info(f"Datasets to process: {config['datasets']}")

    results_dict = {
        "dataset": [],
        "model": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "test_accuracy": [],
        "train_f1": [],
        "val_f1": [],
        "test_f1": [],
    }
    for dataset_name in config["datasets"]:
        # load and prepare data
        data = load_ucr_data(dataset_name, remap_labels_ascending=True)
        train_loader, val_loader, test_loader = prepare_dataloaders(
            data,
            batch_size=train_config["batch_size"],
            val_split=train_config["val_split"],
            seed=train_config["seed"],
        )

        # Set up deterministic training
        logger.debug("Setting up deterministic training...")
        setup_deterministic_training(seed=train_config["seed"])

        # Initialize models
        c_in, c_out = (data.n_features, data.n_classes)
        models_dict = {
            "ResNet": ResNet(c_in, c_out),
            "InceptionTime": InceptionTime(c_in, c_out),
        }
        logger.info(
            f"Initialized models with input channels: {c_in}, output classes: {c_out}"
        )

        for model_name in config["models"]:
            # Check if model exists
            if model_name not in models_dict:
                logger.error(f"Model {model_name} not found in models_dict")
                continue

            # Train model
            model_to_train = models_dict[model_name]
            train_config["model_config"]["dataset_name"] = dataset_name
            model = train_model(
                model=model_to_train,
                train_loader=train_loader,
                val_loader=val_loader,
                config=train_config["model_config"],
            )

            # Evaluate model performance
            logger.info("Evaluating model on train, validation and test set.")
            eval_kwargs = {
                "model": model,
                "device": device,
                "num_classes": data.n_classes,
            }
            train_results = evaluate_model(dataloader=train_loader, **eval_kwargs)
            val_results = evaluate_model(dataloader=val_loader, **eval_kwargs)
            test_results = evaluate_model(dataloader=test_loader, **eval_kwargs)

            # Log evaluation results
            logger.info(
                f"Results for {model_name} on {dataset_name}:\n"
                f"Train - Accuracy: {train_results['accuracy']:.4f}, F1: {train_results['f1']:.4f}\n"
                f"Val   - Accuracy: {val_results['accuracy']:.4f}, F1: {val_results['f1']:.4f}\n"
                f"Test  - Accuracy: {test_results['accuracy']:.4f}, F1: {test_results['f1']:.4f}"
            )

            results_dict["dataset"].append(dataset_name)
            results_dict["model"].append(model_name)
            results_dict["train_accuracy"].append(train_results["accuracy"])
            results_dict["val_accuracy"].append(val_results["accuracy"])
            results_dict["test_accuracy"].append(test_results["accuracy"])
            results_dict["train_f1"].append(train_results["f1"])
            results_dict["val_f1"].append(val_results["f1"])
            results_dict["test_f1"].append(test_results["f1"])

    logger.info("Training pipeline completed. Saving results...")
    results_path = str(MODELS_DIR / "performance.csv")
    pd.DataFrame(results_dict).to_csv(results_path, index=False)
    logger.info("Results saved to file: model_performance.csv")


if __name__ == "__main__":
    main()
