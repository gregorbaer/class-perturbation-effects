# Class-Dependent Perturbation Effects

This repository contains the code and experiment results for our paper:
```
Baer, G., Grau, I., Zhang, C., & Van Gorp, P. (2024). 
Class-Dependent Perturbation Effects in Evaluating Time Series Attributions.
```

## Quick Start (Local Installation)

For a quickly getting started, you can locally install the code base with:
```bash
# Clone repository
git clone https://github.com/gregorbaer/class-perturbation-effects.git
cd class-perturbation-effects

# Quick local installation
pip install -e .
```
If you encounter any installation issues or need a more isolated environment, consider using Docker below.

## Repository Structure

This repository is organized as follows:

### Source Code
- `src/tsxai/`: Core package implementation containing attribution methods, data processing, modeling, and visualization utilities
  - `attribution/`: Implementation of attribution methods and perturbation analysis
  - `data/`: Data loading and preprocessing utilities
  - `modelling/`: Time series classification models and training
  - `utils/`: Helper functions and utilities
  - `visualization/`: Plotting and visualization tools

### Examples and Documentation
- `notebooks/example.ipynb`: Jupyter notebook demonstrating:
  - Core functionality of the code base (tsxai package)
  - How to load and analyze experimental results
  - Reproduction of paper visualizations
  - Example workflows for attribution analysis

### Experiment Scripts
Located in `scripts/`:
- `config.yaml`: Configuration file for experiments
  - Modify parameters to run custom experiments
  - Control settings for attribution methods, perturbation analysis, etc.
- `train_models.py`: Trains time series classifiers
  - Must be run first when using new datasets
  - Generates models for subsequent attribution analysis
- `perturbation_experiment.py`: Main experimental pipeline
  - Generates feature attributions
  - Applies perturbations
  - Calculates correctness metrics and records metadata for each observation
- `aggregate_experiment_results.py`: Results aggregation
  - Creates summary tables shown in paper
  - Computes average DS and class-adjusted DS
  - Aggregates results from perturbation experiments

### Results and Models
- `results/models/`: Trained model weights
  - Contains classifiers used in paper
  - Can be easily loaded via example notebook
  - Includes performance metrics in `performance.csv` (F1, Accuracy)
- `results/perturbation_results/`: Experimental results
  - Raw results from perturbation experiments
  - `results/perturbation_results/paper_results` contains the experiment results included in the paper
    - `compiled_results.csv`: Instance-level results
    - `tables/`: Aggregated results and analysis
  - If you rerun experiments, they will be saved in the `pertubation_results/` dir with the name patter `perturbation_benchmark_<date>_<time>`

### Data
- `data/`: Directory for datasets
  - When using the `load_ucr_data()` function from `tsxai.data.loading` the UCR datasets are automatically downloaded and placed in this directory.

## Docker Installation (Recommended for Reproducibility)
Docker provides an isolated, reproducible environment that works consistently across different systems. Just note that when using the provided docker image instead of local installation, you will not have access to GPU support by default.

### Option 1: Using VSCode with Docker (Preferred Method)
This option provides the best development experience with an integrated environment.

Requirements:
- [Docker](https://docs.docker.com/get-docker/)
- [VSCode](https://code.visualstudio.com/)
- [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) VSCode extension

Steps:
```bash
# Clone repository
git clone https://github.com/gregorbaer/class-perturbation-effects.git
# Open in VSCode
code class-perturbation-effects
```
Then:
1. When VSCode prompts "Reopen in Container", click "Yes"
2. If not prompted:
   - Press F1 (or Ctrl+Shift+P)
   - Type "Remote-Containers: Reopen in Container"
   - Press Enter
   - Note: When building for the first time, this can take a long time

*When to use: This is the recommended method if you:*
- *Want a fully configured development environment*
- *Prefer using VSCode with full IDE capabilities*
- *Want to avoid potential dependency conflicts*

### Option 2: Using Docker Commands
For users who prefer command line or don't use VSCode:

```bash
# Clone repository
git clone https://github.com/gregorbaer/class-perturbation-effects.git
cd class-perturbation-effects

# Build the Docker image
docker build -t class-perturbation-effects .

# Run the container (choose one option):

# Option A: Simple run
docker run -it --rm class-perturbation-effects

# Option B: With mounted volume to persist data
docker run -it --rm -v $(pwd):/app class-perturbation-effects

# Verify installation inside container
python -c "import tsxai; print('TSXAI successfully installed')"
```

*When to use: Choose this if you:*
- *Prefer working from the command line and don't use VSCode*
- *Want to run the code in a clean environment*

### GPU Support (Optional)

While the code works on CPU by default, users with NVIDIA GPUs can enable GPU acceleration in Docker by:

1. Installing NVIDIA Container Toolkit
2. Running Docker with the `--gpus all` flag:
```bash
docker run --gpus all -it --rm -v $(pwd):/app class-perturbation-effects
```
Or for VS Code Dev Containers, add to devcontainer.json:

```json
{
    "runArgs": ["--gpus", "all"]
}
```
Note that GPU support is not strictly necessary and will likely only speed up model training, but not the perturbation experiments by much.

### Troubleshooting

#### Verification
After installation (using any method), verify your setup:
```bash
python -c "import tsxai; print('TSXAI successfully installed')"
```

#### Package Installation
If the regular installation fails, try installing critical dependencies first:
```bash
# Install critical dependencies first
pip install TSInterpret
pip install tsai
# Then the local tsxai package
pip install -e .
```

#### Docker-Specific Issues
If you encounter Docker-related problems:

1. VSCode Docker issues:
   - Check if Docker is running on your system
   - Try "Rebuild Container" from VSCode's command palette
   - Check the "Docker" tab in VSCode for container status

2. Command line Docker issues:
   - Rebuild without cache: `docker build --no-cache -t class-perturbation-effects .`
   - Check logs: `docker logs [container-id]`
   - Ensure Docker daemon is running
   - Try with elevated privileges if needed