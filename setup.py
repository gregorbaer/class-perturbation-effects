from setuptools import find_packages, setup

setup(
    name="tsxai",
    version="0.1",
    description="Code for paper on 'Class-dependent perturbation effects'.",
    author="Gregor Baer",
    author_email="g.baer@tue.nl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "TSInterpret",  # If there are scikit-learn installation errors, install TSInterpret first separately
        "tsai",  # install tsai right after
        "numpy",
        "pandas",
        "torch",
        "lets_plot",
        "tqdm",
        "tabulate",
        "torchmetrics",
        "pytorch_lightning",
        "PyYAML",
        "scikit-learn",
        "ipykernel",
        "ipywidgets",
        "ruff",
    ],
)
