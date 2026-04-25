"""
Utility Functions — Timing, Statistics, and Helpers
=====================================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/monte-carlo-option-pricer

This module provides utility functions used throughout the project:
timing decorators, confidence interval computation, and data export
for the convergence analysis.

WHY THESE MATTER
-----------------
- Timer: Essential for quantifying computational cost. Jane Street cares
  about speed — a pricing engine that takes 10 seconds vs 10 milliseconds
  is the difference between usable and useless in production.
- Confidence Intervals: Monte Carlo estimates are random variables.
  Reporting a price without a confidence interval is incomplete.
  We compute 95% CI using the standard error of the mean.
- Data Export: The convergence analysis generates data that needs to
  be saved for plotting. CSV is the simplest portable format that
  both Python and C can read/write.
"""

import time
import numpy as np
from functools import wraps
from typing import Callable, Tuple, Optional
from pathlib import Path


# ============================================================================
# TIMING UTILITIES
# ============================================================================

class Timer:
    """
    Context manager and decorator for measuring execution time.

    Two ways to use:

    1. Context manager:
       with Timer("Path generation"):
           paths = model.simulate_paths(T, N_paths, N_steps)

    2. Decorator:
       @Timer("Option pricing")
       def price_options(...):
           ...
    """

    def __init__(self, label: str = "Operation") -> None:
        """
        Parameters
        ----------
        label : str
            Description of what's being timed (for display).
        """
        self.label = label
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        """Start timing when entering the context."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        """Stop timing and display when exiting the context."""
        self.elapsed = time.perf_counter() - self.start_time
        print(f"[TIMER] {self.label}: {self.elapsed:.4f} seconds")

    def __call__(self, func: Callable) -> Callable:
        """
        Allow Timer to be used as a decorator.

        Example:
        @Timer("Simulation")
        def run_simulation():
            ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


# ============================================================================
# STATISTICS UTILITIES
# ============================================================================

def confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute the mean and confidence interval of a sample.

    Uses the Central Limit Theorem: for sufficiently large samples,
    the sample mean is approximately normally distributed with
    standard error = std(data) / sqrt(N).

    The (1 - alpha) * 100% confidence interval is:
        mean ± z_{1-alpha/2} * std_error

    For 95% CI, z_{0.975} ≈ 1.96.

    Parameters
    ----------
    data : np.ndarray
        Array of Monte Carlo estimates (e.g., discounted payoffs).
    confidence : float
        Confidence level, default 0.95 for 95% CI.

    Returns
    -------
    mean : float
        Sample mean.
    lower : float
        Lower bound of the confidence interval.
    upper : float
        Upper bound of the confidence interval.

    Examples
    --------
    >>> payoffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> mean, lo, hi = confidence_interval(payoffs)
    >>> print(f"Price: {mean:.4f} [{lo:.4f}, {hi:.4f}]")
    Price: 3.0000 [1.5286, 4.4714]
    """
    N = len(data)
    if N < 2:
        return float(data[0]) if N == 1 else 0.0, 0.0, 0.0

    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(N)

    # z-score for the given confidence level
    # 95% -> 1.96, 99% -> 2.576
    from scipy.stats import norm
    alpha = 1.0 - confidence
    z = norm.ppf(1.0 - alpha / 2.0)

    lower = mean - z * std_err
    upper = mean + z * std_err

    return mean, lower, upper


def relative_error(
    estimate: float,
    true_value: float,
) -> float:
    """
    Compute the relative error of an estimate vs a true value.

    relative_error = |estimate - true_value| / |true_value|

    Parameters
    ----------
    estimate : float
        Monte Carlo estimate.
    true_value : float
        Ground truth value (e.g., Black-Scholes closed-form).

    Returns
    -------
    rel_error : float
        Relative error. Returns inf if true_value is zero.
    """
    if true_value == 0.0:
        return float('inf') if estimate != 0.0 else 0.0
    return abs(estimate - true_value) / abs(true_value)


def absolute_error(estimate: float, true_value: float) -> float:
    """
    Compute the absolute error of an estimate vs a true value.

    Parameters
    ----------
    estimate : float
        Monte Carlo estimate.
    true_value : float
        Ground truth value.

    Returns
    -------
    abs_error : float
        Absolute error.
    """
    return abs(estimate - true_value)


# ============================================================================
# BATCHING — For computing standard errors without storing all payoffs
# ============================================================================

def batch_means(
    payoffs: np.ndarray,
    batch_size: int,
) -> Tuple[float, float]:
    """
    Compute the mean and standard error using the batch means method.

    This is useful for assessing Monte Carlo error when we want to
    avoid the assumption of independence (though in MC simulation,
    paths ARE independent by construction).

    The batch means method:
    1. Divide the N paths into M batches of size batch_size.
    2. Compute the mean of each batch.
    3. The standard error of the overall mean is:
       std(batch_means) / sqrt(M)

    Parameters
    ----------
    payoffs : np.ndarray of shape (N_paths,)
        Discounted payoffs for each path.
    batch_size : int
        Number of paths per batch.

    Returns
    -------
    mean : float
        Overall mean (should match np.mean(payoffs)).
    std_err : float
        Standard error estimated from batch means.
    """
    N = len(payoffs)
    # Truncate to an integer number of batches
    M = N // batch_size
    if M < 2:
        return np.mean(payoffs), np.std(payoffs, ddof=1) / np.sqrt(N)

    truncated = payoffs[: M * batch_size]
    batches = truncated.reshape(M, batch_size)
    batch_means_values = np.mean(batches, axis=1)

    mean = np.mean(batch_means_values)
    std_err = np.std(batch_means_values, ddof=1) / np.sqrt(M)

    return mean, std_err


# ============================================================================
# DATA EXPORT
# ============================================================================

def ensure_dir(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_results_csv(
    filename: str,
    headers: list,
    data: list,
    output_dir: str = "../data",
) -> None:
    """
    Save results to a CSV file for later plotting or analysis.

    Parameters
    ----------
    filename : str
        Name of the CSV file (e.g., 'convergence_N.csv').
    headers : list of str
        Column headers.
    data : list of lists
        Rows of data (each row is a list).
    output_dir : str
        Directory to save the file.
    """
    ensure_dir(output_dir)
    filepath = Path(output_dir) / filename

    with open(filepath, 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in data:
            f.write(','.join(str(x) for x in row) + '\n')

    print(f"[DATA] Saved {len(data)} rows to {filepath}")


# ============================================================================
# SEED MANAGEMENT — Reproducibility across comparisons
# ============================================================================

def generate_seeds(n_seeds: int, base_seed: int = 42) -> list:
    """
    Generate a list of independent seeds for batching.

    When comparing Greek methods, it's critical that they use the
    SAME random paths. The simplest way is to fix one seed for
    path generation, then use separate seeds for each Greek computation
    to ensure independence across methods.

    Parameters
    ----------
    n_seeds : int
        Number of seeds to generate.
    base_seed : int
        Starting seed. Each generated seed is base_seed + i.

    Returns
    -------
    seeds : list of int
        List of seeds.
    """
    return [base_seed + i for i in range(n_seeds)]


# ============================================================================
# PARAMETER VALIDATION
# ============================================================================

def validate_positive(value: float, name: str) -> None:
    """
    Validate that a parameter is positive, raise ValueError if not.

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Parameter name for the error message.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_range(
    value: float, name: str, low: float, high: float
) -> None:
    """
    Validate that a parameter is within [low, high], raise ValueError if not.

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Parameter name for the error message.
    low : float
        Minimum allowed value (inclusive).
    high : float
        Maximum allowed value (inclusive).
    """
    if not (low <= value <= high):
        raise ValueError(
            f"{name} must be in [{low}, {high}], got {value}"
        )