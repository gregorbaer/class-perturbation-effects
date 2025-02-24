from typing import Union

import numpy as np


def _resolve_k(x: np.ndarray, k: Union[int, float]) -> int:
    """
    Resolve k value from either absolute number or percentage.

    Parameters
    ----------
    x : np.ndarray
        Input time series data
    k : Union[int, float]
        If int: absolute number of time steps
        If float: percentage of time series length (0.0 to 1.0)

    Returns
    -------
    int
        Resolved number of time steps
    """
    if isinstance(k, float):
        if not 0 < k <= 1:
            raise ValueError("Percentage k must be between 0 and 1")
        k = max(1, int(k * x.shape[-1]))
    return int(k)


def zero(x: np.ndarray) -> float:
    return 0.0


def mean(x: np.ndarray) -> float:
    return np.mean(x)


def gaussian_noise(x: np.ndarray, n_std: float = 1) -> float:
    return np.random.normal(loc=np.mean(x), scale=n_std * np.std(x))


def opposite(x: np.ndarray) -> np.ndarray:
    return -x


def uniform_noise(x: np.ndarray) -> np.ndarray:
    return np.random.uniform(np.min(x), np.max(x), size=x.shape)


def subsequence_mean(x: np.ndarray, k: Union[int, float]) -> np.ndarray:
    """
    Calculate subsequence mean with flexible k specification.

    Parameters
    ----------
    x : np.ndarray
        Input time series data
    k : Union[int, float]
        If int: use exactly k time steps
        If float: use k% of time series length
    """
    k = _resolve_k(x, k)
    result = np.zeros_like(x)
    for i in range(x.shape[-1]):
        j = max(0, i - k + 1)
        result[..., i] = np.mean(x[..., j : i + 1], axis=-1)
    return result


def inverse(x: np.ndarray) -> np.ndarray:
    return np.max(x) - x


def ood_high(x: np.ndarray) -> float:
    return max(abs(np.max(x)), abs(np.min(x))) * 100


def ood_low(x: np.ndarray) -> float:
    return -max(abs(np.max(x)), abs(np.min(x))) * 100


# Dictionary mapping strategy names to functions
PERTURBATION_FUNCTIONS = {
    "zero": zero,
    "mean": mean,
    "gaussian_noise": gaussian_noise,
    "opposite": opposite,
    "uniform_noise": uniform_noise,
    "subsequence_mean": subsequence_mean,
    "inverse": inverse,
    "ood_high": ood_high,
    "ood_low": ood_low,
}
