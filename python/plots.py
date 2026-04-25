"""
Convergence Analysis and Visualization
========================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/monte-carlo-option-pricer

This module generates all the plots required for the project:

1. CONVERGENCE PLOTS (Error vs Number of Paths N)
   Log-log plot showing the Monte Carlo error decreases as 1/sqrt(N).
   This is the fundamental convergence rate of Monte Carlo.

   We plot:
   - Absolute error |Price_MC(N) - Price_true|
   - Theoretical 1/sqrt(N) reference line
   - 95% confidence bands

2. DISCRETIZATION BIAS PLOTS (Bias vs Time Step dt)
   Euler-Maruyama discretization introduces bias that scales as O(dt).
   This plot shows how the estimated price converges to the true
   continuous-time price as dt -> 0.

3. GREEKS COMPARISON PLOTS
   Bar chart comparing Delta and Vega estimates from:
   - Pathwise method
   - Likelihood ratio method
   - Finite difference
   - Black-Scholes closed-form (ground truth)

WHY THESE MATTER FOR JANE STREET
----------------------------------
Jane Street cares deeply about:
- Convergence rates: How many paths do we need to get 1bp accuracy?
- Bias-variance tradeoff: Euler discretization bias vs Monte Carlo error.
- Greeks accuracy: Which method gives tightest estimates for hedging?

These plots tell the story quantitatively. A good quant can look at
a convergence plot and immediately know if the implementation is correct
(parallel lines on log-log? Check. Slope -1/2? Check. Bias vs dt linear? Check.)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple


# ============================================================================
# GLOBAL STYLE CONFIGURATION
# ============================================================================
# Professional, publication-quality defaults.
# Seaborn-v0_8-darkgrid gives a clean dark grid on white background
# that looks great in both Jupyter notebooks and exported PDFs.
# ============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
})


# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

def _ensure_plots_dir() -> Path:
    """Create and return the plots output directory."""
    plots_dir = Path("../plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


# ============================================================================
# PLOT 1: CONVERGENCE WITH NUMBER OF PATHS (Error vs N)
# ============================================================================

def plot_convergence_N(
    N_values: List[int],
    errors: List[float],
    true_price: float,
    mc_prices: Optional[List[float]] = None,
    std_errors: Optional[List[float]] = None,
    title: str = "Monte Carlo Convergence: Error vs Number of Paths",
    option_type: str = "European Call",
    save: bool = True,
    filename: str = "convergence_N.png",
) -> None:
    """
    Plot the Monte Carlo convergence rate as a function of N.

    This is THE fundamental plot in Monte Carlo methods.
    It demonstrates the 1/sqrt(N) convergence rate.

    The log-log scale is essential:
    - On log-log, error ∝ 1/sqrt(N) = N^{-1/2} becomes a straight line
      with slope -1/2.
    - If the line is straight and parallel to the reference, the
      implementation is correct.

    Parameters
    ----------
    N_values : list of int
        Number of paths used for each estimate.
    errors : list of float
        Absolute error |price_MC - price_true| for each N.
    true_price : float
        Ground truth price (Black-Scholes closed-form).
    mc_prices : list of float, optional
        The actual MC price estimates, for confidence bands.
    std_errors : list of float, optional
        Standard errors for each estimate (for confidence bands).
    title : str
        Plot title.
    option_type : str
        Description of the option being priced.
    save : bool
        If True, save to file.
    filename : str
        Output filename.
    """
    N_values = np.array(N_values, dtype=float)
    errors = np.array(errors)

    fig, ax = plt.subplots(figsize=(10, 6))

    # ------------------------------------------------------------------
    # Plot the empirical error
    # ------------------------------------------------------------------
    ax.loglog(
        N_values, errors, 'o-',
        color='#2196F3',  # Material Blue
        linewidth=1.5,
        markersize=6,
        label=f'Empirical Error ({option_type})',
        zorder=3,
    )

    # ------------------------------------------------------------------
    # Plot the theoretical 1/sqrt(N) reference line
    # Scale it to pass through the first data point
    # ------------------------------------------------------------------
    ref_slope = -0.5
    ref_scale = errors[0] * (N_values[0] ** (-ref_slope))
    N_ref = np.logspace(
        np.log10(N_values[0]),
        np.log10(N_values[-1]),
        100
    )
    ref_line = ref_scale * (N_ref ** ref_slope)

    ax.loglog(
        N_ref, ref_line, '--',
        color='#F44336',  # Material Red
        linewidth=2,
        alpha=0.7,
        label=r'$\propto 1/\sqrt{N}$ (Theoretical)',
        zorder=2,
    )

    # ------------------------------------------------------------------
    # Confidence bands (if standard errors provided)
    # 95% CI ≈ ± 1.96 * std_err
    # ------------------------------------------------------------------
    if mc_prices is not None and std_errors is not None:
        mc_prices = np.array(mc_prices)
        std_errors = np.array(std_errors)

        upper = np.abs(mc_prices + 1.96 * std_errors - true_price)
        lower = np.abs(mc_prices - 1.96 * std_errors - true_price)

        ax.fill_between(
            N_values, lower, upper,
            alpha=0.15,
            color='#2196F3',
            label='95% Confidence Band',
            zorder=1,
        )

    # ------------------------------------------------------------------
    # Labels and formatting
    # ------------------------------------------------------------------
    ax.set_xlabel('Number of Paths (N)')
    ax.set_ylabel('Absolute Error')
    ax.set_title(title)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)

    # Annotate the convergence rate
    ax.text(
        0.05, 0.05,
        r'Error $\propto N^{-1/2}$',
        transform=ax.transAxes,
        fontsize=9,
        color='gray',
        style='italic',
    )

    plt.tight_layout()

    if save:
        filepath = _ensure_plots_dir() / filename
        plt.savefig(filepath, bbox_inches='tight')
        print(f"[PLOT] Saved convergence plot to {filepath}")

    plt.show()


# ============================================================================
# PLOT 2: DISCRETIZATION BIAS (Bias vs dt)
# ============================================================================

def plot_discretization_bias(
    dt_values: List[float],
    prices: List[float],
    true_price: float,
    title: str = "Euler-Maruyama Discretization Bias vs Time Step",
    option_type: str = "European Call (Heston)",
    save: bool = True,
    filename: str = "convergence_dt.png",
) -> None:
    """
    Plot the discretization bias of the Euler-Maruyama scheme.

    For the Heston model, Euler-Maruyama introduces a bias that
    scales approximately linearly with dt:
        Price(dt) ≈ Price_true + C * dt

    As dt -> 0, the discrete approximation converges to the true
    continuous-time price.

    This plot is on a LOG-LINEAR scale (log dt on x-axis, bias on y)
    to show the linear convergence.

    Parameters
    ----------
    dt_values : list of float
        Time step sizes used.
    prices : list of float
        MC price estimate for each dt (using enough paths that
        MC error is negligible compared to discretization bias).
    true_price : float
        Reference price (from very small dt or closed-form).
    title : str
        Plot title.
    option_type : str
        Description of the option.
    save : bool
        If True, save to file.
    filename : str
        Output filename.
    """
    dt_values = np.array(dt_values)
    prices = np.array(prices)
    biases = prices - true_price

    fig, ax = plt.subplots(figsize=(10, 6))

    # ------------------------------------------------------------------
    # Plot bias vs dt (log-linear: dt is logged, bias is linear)
    # ------------------------------------------------------------------
    ax.semilogx(
        dt_values, biases, 'o-',
        color='#4CAF50',  # Material Green
        linewidth=1.5,
        markersize=8,
        label=f'Bias ({option_type})',
        zorder=3,
    )

    # ------------------------------------------------------------------
    # Zero-bias reference line
    # ------------------------------------------------------------------
    ax.axhline(
        y=0, color='black', linewidth=0.8,
        linestyle='--', alpha=0.5,
        label='Zero Bias (True Price)',
    )

    # ------------------------------------------------------------------
    # Linear reference: bias = C * dt
    # Fit a line through the data to show approximate O(dt) scaling
    # ------------------------------------------------------------------
    # On semilogx, O(dt) appears as an exponential curve.
    # We plot a reference line: bias ≈ C * dt
    # Fit: bias = slope * dt
    slope = np.polyfit(dt_values, biases, 1)[0]
    dt_ref = np.logspace(np.log10(dt_values[-1]), np.log10(dt_values[0]), 100)
    bias_ref = slope * dt_ref

    ax.semilogx(
        dt_ref, bias_ref, ':',
        color='#FF9800',  # Material Orange
        linewidth=2,
        alpha=0.7,
        label=r'$\propto \Delta t$ (Reference)',
        zorder=2,
    )

    # ------------------------------------------------------------------
    # Labels and formatting
    # ------------------------------------------------------------------
    ax.set_xlabel(r'Time Step ($\Delta t$)')
    ax.set_ylabel('Bias (Price - True Price)')
    ax.set_title(title)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)

    # Annotate
    ax.text(
        0.95, 0.05,
        r'Bias $\propto \Delta t$ (Euler-Maruyama)',
        transform=ax.transAxes,
        fontsize=9,
        color='gray',
        style='italic',
        ha='right',
    )

    # Invert x-axis so smaller dt is on the right
    ax.invert_xaxis()

    plt.tight_layout()

    if save:
        filepath = _ensure_plots_dir() / filename
        plt.savefig(filepath, bbox_inches='tight')
        print(f"[PLOT] Saved discretization bias plot to {filepath}")

    plt.show()


# ============================================================================
# PLOT 3: GREEKS COMPARISON
# ============================================================================

def plot_greeks_comparison(
    greek_name: str,
    pathwise_value: float,
    lr_value: float,
    fd_value: float,
    true_value: float,
    pathwise_std: Optional[float] = None,
    lr_std: Optional[float] = None,
    title: Optional[str] = None,
    save: bool = True,
    filename: Optional[str] = None,
) -> None:
    """
    Bar chart comparing Greek estimates from different methods.

    This is the key visualization for the Greeks section.
    It shows:
    - Pathwise: Low variance but fails for discontinuous payoffs.
    - Likelihood Ratio: Higher variance (wider error bars) but works
      for ALL options, including barriers.
    - Finite Difference: Baseline, expensive (2x simulation cost).
    - True Value: Black-Scholes closed-form (for European options).

    Parameters
    ----------
    greek_name : str
        Name of the Greek ('Delta' or 'Vega').
    pathwise_value : float
        Pathwise estimate.
    lr_value : float
        Likelihood ratio estimate.
    fd_value : float
        Finite difference estimate.
    true_value : float
        Closed-form true value.
    pathwise_std : float, optional
        Standard error of pathwise estimate.
    lr_std : float, optional
        Standard error of likelihood ratio estimate.
    title : str, optional
        Plot title.
    save : bool
        If True, save to file.
    filename : str, optional
        Output filename.
    """
    methods = ['Pathwise', 'Likelihood\nRatio', 'Finite\nDiff', 'True\n(BS)']
    values = [pathwise_value, lr_value, fd_value, true_value]
    errors = [pathwise_std or 0, lr_std or 0, 0, 0]

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9E9E9E']

    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(methods))
    bars = ax.bar(
        x, values, width=0.5,
        color=colors,
        edgecolor='black',
        linewidth=0.8,
        yerr=errors,
        capsize=8,
        error_kw={'linewidth': 1.5},
    )

    # ------------------------------------------------------------------
    # Highlight the true value with a horizontal line
    # ------------------------------------------------------------------
    ax.axhline(
        y=true_value,
        color='#9E9E9E',
        linewidth=1.5,
        linestyle='--',
        alpha=0.7,
        label=f'True Value = {true_value:.6f}',
    )

    # ------------------------------------------------------------------
    # Add value labels on top of each bar
    # ------------------------------------------------------------------
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02 * abs(height),
            f'{val:.6f}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
        )

    # ------------------------------------------------------------------
    # Labels and formatting
    # ------------------------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel(f'{greek_name} Value')
    ax.set_title(
        title or f'{greek_name} Comparison: Pathwise vs Likelihood Ratio vs Finite Diff'
    )
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save:
        fname = filename or f'{greek_name.lower()}_comparison.png'
        filepath = _ensure_plots_dir() / fname
        plt.savefig(filepath, bbox_inches='tight')
        print(f"[PLOT] Saved {greek_name} comparison to {filepath}")

    plt.show()


# ============================================================================
# PLOT 4: VARIANCE COMPARISON (Pathwise vs Likelihood Ratio)
# ============================================================================

def plot_variance_comparison(
    N_values: List[int],
    pathwise_var: List[float],
    lr_var: List[float],
    greek_name: str = "Delta",
    title: Optional[str] = None,
    save: bool = True,
    filename: Optional[str] = None,
) -> None:
    """
    Plot the variance of pathwise vs likelihood ratio as N increases.

    This demonstrates the KEY result: likelihood ratio has higher
    variance than pathwise, but both converge at 1/N rate
    (variance of the estimator ∝ 1/N).

    Parameters
    ----------
    N_values : list of int
        Number of paths.
    pathwise_var : list of float
        Variance of pathwise estimator for each N.
    lr_var : list of float
        Variance of likelihood ratio estimator for each N.
    greek_name : str
        Name of the Greek.
    title : str, optional
        Plot title.
    save : bool
        If True, save to file.
    filename : str, optional
        Output filename.
    """
    N_values = np.array(N_values, dtype=float)
    pathwise_var = np.array(pathwise_var)
    lr_var = np.array(lr_var)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(
        N_values, pathwise_var, 'o-',
        color='#2196F3',
        linewidth=1.5,
        markersize=6,
        label='Pathwise Variance',
        zorder=3,
    )

    ax.loglog(
        N_values, lr_var, 's-',
        color='#FF9800',
        linewidth=1.5,
        markersize=6,
        label='Likelihood Ratio Variance',
        zorder=3,
    )

    # 1/N reference line
    ref_slope = -1.0
    ref_scale = pathwise_var[0] * (N_values[0] ** (-ref_slope))
    N_ref = np.logspace(
        np.log10(N_values[0]),
        np.log10(N_values[-1]),
        100
    )
    ref_line = ref_scale * (N_ref ** ref_slope)

    ax.loglog(
        N_ref, ref_line, '--',
        color='gray',
        linewidth=2,
        alpha=0.5,
        label=r'$\propto 1/N$ (Reference)',
        zorder=1,
    )

    ax.set_xlabel('Number of Paths (N)')
    ax.set_ylabel('Estimator Variance')
    ax.set_title(
        title or f'Variance Comparison: Pathwise vs Likelihood Ratio ({greek_name})'
    )
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()

    if save:
        fname = filename or f'{greek_name.lower()}_variance.png'
        filepath = _ensure_plots_dir() / fname
        plt.savefig(filepath, bbox_inches='tight')
        print(f"[PLOT] Saved variance comparison to {filepath}")

    plt.show()


# ============================================================================
# PLOT 5: ERROR vs N FOR GREEKS
# ============================================================================

def plot_greeks_convergence(
    N_values: List[int],
    pathwise_errors: List[float],
    lr_errors: List[float],
    greek_name: str = "Delta",
    title: Optional[str] = None,
    save: bool = True,
    filename: Optional[str] = None,
) -> None:
    """
    Log-log plot of Greek error vs N for both pathwise and likelihood ratio.

    Both should converge at 1/sqrt(N), but likelihood ratio will have
    a larger constant factor (higher variance).

    Parameters
    ----------
    N_values : list of int
        Number of paths.
    pathwise_errors : list of float
        Absolute error of pathwise estimate for each N.
    lr_errors : list of float
        Absolute error of likelihood ratio estimate for each N.
    greek_name : str
        Name of the Greek.
    title : str, optional
        Plot title.
    save : bool
        If True, save to file.
    filename : str, optional
        Output filename.
    """
    N_values = np.array(N_values, dtype=float)
    pathwise_errors = np.array(pathwise_errors)
    lr_errors = np.array(lr_errors)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(
        N_values, pathwise_errors, 'o-',
        color='#2196F3',
        linewidth=1.5,
        markersize=6,
        label='Pathwise Error',
        zorder=3,
    )

    ax.loglog(
        N_values, lr_errors, 's-',
        color='#FF9800',
        linewidth=1.5,
        markersize=6,
        label='Likelihood Ratio Error',
        zorder=3,
    )

    # 1/sqrt(N) reference
    ref_slope = -0.5
    ref_scale = pathwise_errors[0] * (N_values[0] ** (-ref_slope))
    N_ref = np.logspace(
        np.log10(N_values[0]),
        np.log10(N_values[-1]),
        100
    )
    ref_line = ref_scale * (N_ref ** ref_slope)

    ax.loglog(
        N_ref, ref_line, '--',
        color='gray',
        linewidth=2,
        alpha=0.5,
        label=r'$\propto 1/\sqrt{N}$',
        zorder=1,
    )

    ax.set_xlabel('Number of Paths (N)')
    ax.set_ylabel('Absolute Error')
    ax.set_title(
        title or f'{greek_name} Convergence: Pathwise vs Likelihood Ratio'
    )
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()

    if save:
        fname = filename or f'{greek_name.lower()}_convergence.png'
        filepath = _ensure_plots_dir() / fname
        plt.savefig(filepath, bbox_inches='tight')
        print(f"[PLOT] Saved Greeks convergence to {filepath}")

    plt.show()