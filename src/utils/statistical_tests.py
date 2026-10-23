"""
Statistical testing utilities for model evaluation.
"""
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def compute_statistics(
    values: List[float], confidence: float = 0.95
) -> Dict[str, float]:
    """
    Compute mean, std, and confidence interval.

    Args:
        values: List of metric values from multiple runs
        confidence: Confidence level for interval (default: 0.95)

    Returns:
        Dictionary with mean, std, ci_lower, ci_upper
    """
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std
    se = std / np.sqrt(n)

    # Compute confidence interval using t-distribution
    t_value = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci_lower = mean - t_value * se
    ci_upper = mean + t_value * se

    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def paired_t_test(
    values_a: List[float], values_b: List[float]
) -> Dict[str, float]:
    """
    Perform paired t-test between two models.

    Args:
        values_a: Metrics from model A (multiple seeds)
        values_b: Metrics from model B (multiple seeds)

    Returns:
        Dictionary with t_statistic, p_value, and effect_size
    """
    values_a = np.array(values_a)
    values_b = np.array(values_b)

    if len(values_a) != len(values_b):
        raise ValueError("Arrays must have the same length for paired test")

    t_stat, p_value = stats.ttest_rel(values_a, values_b)

    # Cohen's d for effect size
    diff = values_a - values_b
    effect_size = np.mean(diff) / np.std(diff, ddof=1)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "effect_size": effect_size,
    }


def wilcoxon_test(
    values_a: List[float], values_b: List[float]
) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Args:
        values_a: Metrics from model A
        values_b: Metrics from model B

    Returns:
        Dictionary with statistic and p_value
    """
    values_a = np.array(values_a)
    values_b = np.array(values_b)

    if len(values_a) != len(values_b):
        raise ValueError("Arrays must have the same length")

    statistic, p_value = stats.wilcoxon(values_a, values_b)

    return {
        "statistic": statistic,
        "p_value": p_value,
    }


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values from multiple tests

    Returns:
        List of corrected p-values
    """
    n_tests = len(p_values)
    return [min(p * n_tests, 1.0) for p in p_values]


def format_results_with_ci(
    mean: float, std: float, ci_lower: float, ci_upper: float
) -> str:
    """
    Format results as: mean ± std [ci_lower, ci_upper]

    Args:
        mean: Mean value
        std: Standard deviation
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval

    Returns:
        Formatted string
    """
    return f"{mean:.4f} ± {std:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"


def evaluate_with_multiple_seeds(
    evaluate_fn,
    seeds: List[int],
    metric_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Run evaluation with multiple random seeds and compute statistics.

    Args:
        evaluate_fn: Function that takes a seed and returns metrics dict
        seeds: List of random seeds
        metric_names: Names of metrics to track

    Returns:
        Dictionary mapping metric names to their statistics
    """
    results = {metric: [] for metric in metric_names}

    for seed in seeds:
        metrics = evaluate_fn(seed)
        for metric in metric_names:
            if metric in metrics:
                results[metric].append(metrics[metric])

    statistics = {}
    for metric, values in results.items():
        if len(values) > 0:
            statistics[metric] = compute_statistics(values)

    return statistics
