"""
Enhanced statistical analysis tools for reviewer responses.
Addresses Reviewer #4's requirements for rigorous statistical reporting.
"""
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pandas as pd


def compute_comprehensive_statistics(
    values: List[float],
    confidence: float = 0.95,
    check_normality: bool = True
) -> Dict[str, float]:
    """
    Compute comprehensive statistics including mean, std, CI, and normality test.

    Args:
        values: List of metric values from multiple runs
        confidence: Confidence level for interval (default: 0.95)
        check_normality: Whether to perform Shapiro-Wilk test

    Returns:
        Dictionary with complete statistics
    """
    values = np.array(values)
    n = len(values)

    if n < 2:
        raise ValueError("Need at least 2 values for statistical analysis")

    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std
    se = std / np.sqrt(n)

    # Compute confidence interval using t-distribution
    t_value = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci_lower = mean - t_value * se
    ci_upper = mean + t_value * se

    result = {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "min": np.min(values),
        "max": np.max(values),
        "median": np.median(values),
    }

    # Normality test (Shapiro-Wilk)
    if check_normality and n >= 3:
        stat, p_value = stats.shapiro(values)
        result["shapiro_stat"] = stat
        result["shapiro_pvalue"] = p_value
        result["is_normal"] = p_value > 0.05

    return result


def paired_comparison_with_effect_size(
    values_a: List[float],
    values_b: List[float],
    method: str = "auto",
    alternative: str = "two-sided"
) -> Dict[str, float]:
    """
    Perform paired statistical comparison with effect size.

    Args:
        values_a: Metrics from model A (multiple seeds)
        values_b: Metrics from model B (multiple seeds)
        method: "ttest", "wilcoxon", or "auto" (chooses based on normality)
        alternative: "two-sided", "greater", or "less"

    Returns:
        Dictionary with test results and effect size
    """
    values_a = np.array(values_a)
    values_b = np.array(values_b)

    if len(values_a) != len(values_b):
        raise ValueError("Arrays must have the same length for paired test")

    n = len(values_a)
    diff = values_a - values_b

    # Check normality of differences
    if method == "auto" and n >= 3:
        _, p_normal = stats.shapiro(diff)
        use_parametric = p_normal > 0.05
    else:
        use_parametric = method == "ttest"

    # Perform appropriate test
    if use_parametric:
        t_stat, p_value = stats.ttest_rel(values_a, values_b, alternative=alternative)
        test_name = "paired_t_test"
        statistic = t_stat
    else:
        w_stat, p_value = stats.wilcoxon(values_a, values_b, alternative=alternative)
        test_name = "wilcoxon_signed_rank"
        statistic = w_stat

    # Cohen's d for effect size
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    return {
        "test_name": test_name,
        "statistic": statistic,
        "p_value": p_value,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "cohens_d": cohens_d,
        "effect_size": effect_interpretation,
        "n_pairs": n,
    }


def multiple_comparison_correction(
    p_values: List[float],
    method: str = "bonferroni",
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Apply multiple comparison correction.

    Args:
        p_values: List of p-values from multiple tests
        method: "bonferroni", "holm", or "fdr_bh" (Benjamini-Hochberg)
        alpha: Significance level

    Returns:
        Dictionary with corrected p-values and significance
    """
    p_values = np.array(p_values)
    n_tests = len(p_values)

    if method == "bonferroni":
        corrected = np.minimum(p_values * n_tests, 1.0)
        threshold = alpha / n_tests

    elif method == "holm":
        # Holm-Bonferroni method
        order = np.argsort(p_values)
        corrected = np.zeros_like(p_values)
        for i, idx in enumerate(order):
            corrected[idx] = min(p_values[idx] * (n_tests - i), 1.0)
            if i > 0:
                corrected[idx] = max(corrected[idx], corrected[order[i-1]])
        threshold = alpha / n_tests

    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR control
        order = np.argsort(p_values)
        corrected = np.zeros_like(p_values)
        for i, idx in enumerate(order):
            corrected[idx] = min(p_values[idx] * n_tests / (i + 1), 1.0)
            if i > 0:
                corrected[idx] = max(corrected[idx], corrected[order[i-1]])
        threshold = alpha
    else:
        raise ValueError(f"Unknown correction method: {method}")

    significant = corrected < alpha

    return {
        "method": method,
        "alpha": alpha,
        "n_tests": n_tests,
        "corrected_alpha": threshold,
        "raw_p_values": p_values.tolist(),
        "corrected_p_values": corrected.tolist(),
        "significant": significant.tolist(),
        "n_significant": int(np.sum(significant)),
    }


def compare_multiple_methods(
    results_dict: Dict[str, Dict[str, List[float]]],
    baseline_name: str,
    metric_names: List[str],
    correction_method: str = "bonferroni",
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compare multiple methods against a baseline across multiple metrics.

    Args:
        results_dict: {method_name: {metric_name: [values...]}}
        baseline_name: Name of baseline method
        metric_names: List of metrics to compare
        correction_method: Multiple comparison correction method
        alpha: Significance level

    Returns:
        DataFrame with comparison results
    """
    if baseline_name not in results_dict:
        raise ValueError(f"Baseline {baseline_name} not found in results")

    baseline_results = results_dict[baseline_name]
    comparison_results = []
    all_p_values = []

    for method_name, method_results in results_dict.items():
        if method_name == baseline_name:
            continue

        for metric in metric_names:
            if metric not in baseline_results or metric not in method_results:
                continue

            baseline_values = baseline_results[metric]
            method_values = method_results[metric]

            # Perform comparison
            comp = paired_comparison_with_effect_size(
                method_values, baseline_values, method="auto"
            )

            comparison_results.append({
                "method": method_name,
                "metric": metric,
                "baseline_mean": np.mean(baseline_values),
                "baseline_std": np.std(baseline_values, ddof=1),
                "method_mean": np.mean(method_values),
                "method_std": np.std(method_values, ddof=1),
                "mean_diff": comp["mean_diff"],
                "test": comp["test_name"],
                "statistic": comp["statistic"],
                "p_value": comp["p_value"],
                "cohens_d": comp["cohens_d"],
                "effect_size": comp["effect_size"],
            })

            all_p_values.append(comp["p_value"])

    # Apply multiple comparison correction
    if len(all_p_values) > 0:
        correction = multiple_comparison_correction(
            all_p_values, method=correction_method, alpha=alpha
        )

        for i, result in enumerate(comparison_results):
            result["p_value_corrected"] = correction["corrected_p_values"][i]
            result["significant"] = correction["significant"][i]

            # Add significance markers
            p_corr = result["p_value_corrected"]
            if p_corr < 0.001:
                result["sig_marker"] = "***"
            elif p_corr < 0.01:
                result["sig_marker"] = "**"
            elif p_corr < 0.05:
                result["sig_marker"] = "*"
            else:
                result["sig_marker"] = "ns"

    df = pd.DataFrame(comparison_results)
    return df


def format_result_with_stats(
    mean: float,
    std: float,
    p_value: Optional[float] = None,
    format_str: str = ".4f"
) -> str:
    """
    Format result as: mean ± std with optional significance marker.

    Args:
        mean: Mean value
        std: Standard deviation
        p_value: Optional p-value for significance marking
        format_str: Format string for numbers

    Returns:
        Formatted string like "0.9356 ± 0.0023**"
    """
    result = f"{mean:{format_str}} ± {std:{format_str}}"

    if p_value is not None:
        if p_value < 0.001:
            result += "***"
        elif p_value < 0.01:
            result += "**"
        elif p_value < 0.05:
            result += "*"

    return result


def generate_latex_table_with_stats(
    results_dict: Dict[str, Dict[str, List[float]]],
    metric_names: List[str],
    method_order: List[str],
    baseline_name: str,
    caption: str,
    label: str,
    correction_method: str = "bonferroni"
) -> str:
    """
    Generate LaTeX table with statistical comparison.

    Args:
        results_dict: {method_name: {metric_name: [values...]}}
        metric_names: Metrics to include in table
        method_order: Order of methods in rows
        baseline_name: Baseline for comparison
        caption: Table caption
        label: LaTeX label
        correction_method: Multiple comparison correction

    Returns:
        LaTeX table code
    """
    # Compute comparisons
    comparison_df = compare_multiple_methods(
        results_dict, baseline_name, metric_names, correction_method
    )

    # Build LaTeX table
    n_metrics = len(metric_names)
    col_spec = "l" + "c" * n_metrics

    latex = f"\\begin{{table}}[htbp]\n"
    latex += f"\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex += "\\toprule\n"

    # Header
    header = "Method & " + " & ".join(metric_names) + " \\\\\n"
    latex += header
    latex += "\\midrule\n"

    # Data rows
    for method_name in method_order:
        if method_name not in results_dict:
            continue

        row = method_name
        for metric in metric_names:
            if metric not in results_dict[method_name]:
                row += " & -"
                continue

            values = results_dict[method_name][metric]
            mean = np.mean(values)
            std = np.std(values, ddof=1)

            # Get p-value if not baseline
            p_value = None
            if method_name != baseline_name:
                comp_row = comparison_df[
                    (comparison_df["method"] == method_name) &
                    (comparison_df["metric"] == metric)
                ]
                if not comp_row.empty:
                    p_value = comp_row.iloc[0]["p_value_corrected"]

            formatted = format_result_with_stats(mean, std, p_value)
            row += f" & {formatted}"

        row += " \\\\\n"
        latex += row

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"

    # Add footnote explaining significance markers
    latex += "\\begin{tablenotes}\n"
    latex += "\\small\n"
    latex += "\\item Values are mean $\\pm$ standard deviation over 5 random seeds.\n"
    latex += f"\\item Significance markers: ***$p<0.001$, **$p<0.01$, *$p<0.05$ "
    latex += f"(paired comparison vs. {baseline_name}, {correction_method} correction).\n"
    latex += "\\end{tablenotes}\n"
    latex += "\\end{table}\n"

    return latex


def power_analysis(
    effect_size: float,
    n_samples: int,
    alpha: float = 0.05,
    power: float = 0.8
) -> Dict[str, float]:
    """
    Simple power analysis for paired t-test.

    Args:
        effect_size: Cohen's d
        n_samples: Number of paired samples
        alpha: Significance level
        power: Desired statistical power

    Returns:
        Dictionary with power analysis results
    """
    from scipy.stats import t as t_dist

    # Critical t-value for two-tailed test
    df = n_samples - 1
    t_crit = t_dist.ppf(1 - alpha/2, df)

    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n_samples)

    # Calculate achieved power
    achieved_power = 1 - t_dist.cdf(t_crit, df, ncp) + t_dist.cdf(-t_crit, df, ncp)

    # Calculate required sample size for desired power
    # This is approximate
    required_n = max(int(np.ceil((1.96 + 0.84) ** 2 / (effect_size ** 2))), 3)

    return {
        "effect_size": effect_size,
        "current_n": n_samples,
        "alpha": alpha,
        "achieved_power": achieved_power,
        "desired_power": power,
        "sufficient_power": achieved_power >= power,
        "required_n_for_power": required_n,
    }


def bootstrap_confidence_interval(
    values: np.ndarray,
    statistic_fn=np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for any statistic.

    Args:
        values: Data array
        statistic_fn: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed

    Returns:
        Tuple of (statistic, lower_ci, upper_ci)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(values)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_stats[i] = statistic_fn(sample)

    # Compute percentile confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    statistic = statistic_fn(values)

    return statistic, ci_lower, ci_upper


if __name__ == "__main__":
    # Example usage
    print("Enhanced Statistical Analysis Tools - Example Usage\n")

    # Simulate some results
    np.random.seed(42)
    method_a = np.random.normal(0.935, 0.002, 5)  # MHGNano
    method_b = np.random.normal(0.901, 0.004, 5)  # Baseline

    print("1. Comprehensive Statistics:")
    stats_a = compute_comprehensive_statistics(method_a)
    print(f"   Method A: {stats_a['mean']:.4f} ± {stats_a['std']:.4f}")
    print(f"   95% CI: [{stats_a['ci_lower']:.4f}, {stats_a['ci_upper']:.4f}]")
    print(f"   Normal: {stats_a.get('is_normal', 'N/A')}\n")

    print("2. Paired Comparison:")
    comparison = paired_comparison_with_effect_size(method_a, method_b)
    print(f"   Test: {comparison['test_name']}")
    print(f"   p-value: {comparison['p_value']:.6f}")
    print(f"   Cohen's d: {comparison['cohens_d']:.4f} ({comparison['effect_size']})\n")

    print("3. Power Analysis:")
    power = power_analysis(comparison['cohens_d'], len(method_a))
    print(f"   Achieved power: {power['achieved_power']:.4f}")
    print(f"   Sufficient: {power['sufficient_power']}")
    print(f"   Required n: {power['required_n_for_power']}\n")

    print("4. Formatted Result:")
    formatted = format_result_with_stats(
        stats_a['mean'], stats_a['std'], comparison['p_value']
    )
    print(f"   {formatted}")
