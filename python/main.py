"""
Monte Carlo Option Pricer — Main Entry Point
==============================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/monte-carlo-option-pricer

This is the main entry point that orchestrates the entire project:
1. Black-Scholes European option pricing and validation.
2. Heston model simulation with Euler-Maruyama.
3. Asian and Barrier option pricing.
4. Greeks computation (pathwise, likelihood ratio, finite difference).
5. Convergence analysis and plotting.

RUNNING THE FULL PIPELINE
---------------------------
    python main.py

This will:
- Price a European call under Black-Scholes and compare with closed-form.
- Price Asian and Barrier options under Black-Scholes.
- Price all three under Heston.
- Compute Delta and Vega using all methods.
- Generate convergence plots (error vs N, bias vs dt).
- Generate Greeks comparison plots.
- Save results to ../data/ and plots to ../plots/.

CONFIGURATION
--------------
All parameters are defined in the config dictionary at the bottom of
this file. Modify them to experiment with different scenarios.

PERFORMANCE NOTES
------------------
- The convergence study uses up to 2^20 (~1M) paths. On a modern CPU,
  this takes ~5-10 seconds for GBM, ~30-60 seconds for Heston.
- Reduce max_paths during development to iterate faster.
- All simulations are vectorized with NumPy — no Python loops over paths.
"""

import numpy as np
import sys
from pathlib import Path

# Add the local directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from models import GBMPath, HestonPath, set_seed
from options import EuropeanOption, AsianOption, BarrierOption
from greeks import (
    PathwiseGreeks,
    LikelihoodRatioGreeks,
    compute_all_deltas,
    compute_all_vegas,
    black_scholes_price,
    black_scholes_delta,
    black_scholes_vega,
    finite_difference_delta,
    finite_difference_vega,
)
from utils import (
    Timer,
    confidence_interval,
    relative_error,
    save_results_csv,
    ensure_dir,
)
from plots import (
    plot_convergence_N,
    plot_discretization_bias,
    plot_greeks_comparison,
    plot_variance_comparison,
    plot_greeks_convergence,
)


# ============================================================================
# CONFIGURATION — Modify these parameters to experiment
# ============================================================================

config = {
    # ------------------------------------------------------------------
    # Asset and Market Parameters
    # ------------------------------------------------------------------
    "S0": 100.0,        # Initial asset price
    "K": 100.0,         # Strike price (ATM: K = S0)
    "T": 1.0,           # Time to maturity in years
    "r": 0.05,          # Risk-free rate (5%)
    "sigma": 0.20,      # Volatility for Black-Scholes (20%)

    # ------------------------------------------------------------------
    # Heston Model Parameters
    # ------------------------------------------------------------------
    "v0": 0.04,         # Initial variance (sigma_0^2 = 0.20^2 = 0.04)
    "kappa": 2.0,       # Mean-reversion speed
    "theta": 0.04,      # Long-run variance
    "xi": 0.30,         # Volatility of variance ("vol of vol")
    "rho": -0.70,       # Correlation (leverage effect)

    # ------------------------------------------------------------------
    # Barrier Option
    # ------------------------------------------------------------------
    "B": 120.0,         # Barrier level (> S0 for up-and-out)

    # ------------------------------------------------------------------
    # Simulation Parameters
    # ------------------------------------------------------------------
    "base_seed": 42,                        # Seed for reproducibility
    "N_paths_pricing": 500_000,             # Paths for standard pricing
    "N_steps_pricing": 252,                 # Daily steps for 1 year
    "N_paths_convergence": [                # Path counts for convergence study
        1000, 2000, 5000, 10000, 20000,
        50000, 100000, 200000, 500000,
    ],
    "N_steps_discretization": [             # Step counts for bias study
        4, 8, 16, 32, 64, 128, 252,
    ],
    "fd_h": 1e-4,                           # Perturbation for finite diff
}


# ============================================================================
# SECTION 1: Black-Scholes European Option — Validation
# ============================================================================

def run_black_scholes_validation() -> None:
    """
    Price a European call under Black-Scholes and validate against
    the closed-form formula.

    This is the "Hello World" of the project. If this doesn't match,
    nothing else will.
    """
    print("\n" + "=" * 72)
    print("SECTION 1: Black-Scholes European Call — Validation")
    print("=" * 72)

    S0 = config["S0"]
    K = config["K"]
    T = config["T"]
    r = config["r"]
    sigma = config["sigma"]
    N = config["N_paths_pricing"]

    # Closed-form Black-Scholes price
    bs_price = black_scholes_price(S0, K, T, r, sigma)
    bs_delta = black_scholes_delta(S0, K, T, r, sigma)
    bs_vega = black_scholes_vega(S0, K, T, r, sigma)

    print(f"\nBlack-Scholes Closed-Form:")
    print(f"  Price: {bs_price:.6f}")
    print(f"  Delta: {bs_delta:.6f}")
    print(f"  Vega:  {bs_vega:.6f}")

    # Monte Carlo simulation
    set_seed(config["base_seed"])

    gbm = GBMPath(S0, r, sigma)
    option = EuropeanOption(K, T, r)

    with Timer("GBM Terminal Simulation"):
        S_T = gbm.simulate_terminal(T, N)

    mc_price = option.price_from_terminal(S_T)
    discounted_payoffs = option._discount * option.payoff(S_T)

    mean, lo, hi = confidence_interval(discounted_payoffs)
    rel_err = relative_error(mc_price, bs_price)

    print(f"\nMonte Carlo (N={N:,}):")
    print(f"  Price: {mc_price:.6f} [{lo:.6f}, {hi:.6f}] 95% CI")
    print(f"  Relative Error vs BS: {rel_err:.6%}")

    # ------------------------------------------------------------------
    # Delta and Vega via all methods
    # ------------------------------------------------------------------
    print(f"\nGreeks Comparison (N={N:,}):")

    # Delta
    delta_results = compute_all_deltas(
        S_T=S_T,
        payoffs=option.payoff(S_T),
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        discount_factor=option._discount,
        price_func=lambda s: EuropeanOption(K, T, r).price_from_terminal(
            GBMPath(s, r, sigma).simulate_terminal(T, N, seed=config["base_seed"])
        ),
        fd_h=config["fd_h"],
    )

    print(f"\n  Delta:")
    for method, value in delta_results.items():
        err = relative_error(value, bs_delta)
        print(f"    {method:<20s}: {value:.6f}  (err: {err:.6%})")

    # Vega
    vega_results = compute_all_vegas(
        S_T=S_T,
        payoffs=option.payoff(S_T),
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        discount_factor=option._discount,
        price_func=lambda s: EuropeanOption(K, T, r).price_from_terminal(
            GBMPath(S0, r, s).simulate_terminal(T, N, seed=config["base_seed"])
        ),
        fd_h=config["fd_h"],
    )

    print(f"\n  Vega:")
    for method, value in vega_results.items():
        err = relative_error(value, bs_vega)
        print(f"    {method:<20s}: {value:.6f}  (err: {err:.6%})")

    return delta_results, vega_results


# ============================================================================
# SECTION 2: Path-Dependent Options under Black-Scholes
# ============================================================================

def run_path_dependent_options() -> None:
    """
    Price Asian and Barrier options under Black-Scholes.

    Asian options need the full path for the arithmetic average.
    Barrier options need the full path to monitor the knockout condition.
    """
    print("\n" + "=" * 72)
    print("SECTION 2: Path-Dependent Options under Black-Scholes")
    print("=" * 72)

    S0 = config["S0"]
    K = config["K"]
    T = config["T"]
    r = config["r"]
    sigma = config["sigma"]
    B = config["B"]
    N = config["N_paths_pricing"]
    N_steps = config["N_steps_pricing"]

    set_seed(config["base_seed"])

    gbm = GBMPath(S0, r, sigma)

    with Timer("GBM Full Path Simulation"):
        paths = gbm.simulate_paths(T, N, N_steps)

    # ------------------------------------------------------------------
    # Asian Option
    # ------------------------------------------------------------------
    asian = AsianOption(K, T, r)

    with Timer("Asian Option Pricing"):
        asian_price = asian.price(paths)

    asian_payoffs = asian._discount * asian.payoff(paths)
    asian_mean, asian_lo, asian_hi = confidence_interval(asian_payoffs)

    print(f"\nAsian Call (Arithmetic Average, N={N:,}, steps={N_steps}):")
    print(f"  Price: {asian_price:.6f} [{asian_lo:.6f}, {asian_hi:.6f}] 95% CI")

    # Pathwise Delta for Asian
    pw_asian_delta = PathwiseGreeks.delta_asian(paths, S0, K, T, r)
    asian_delta = asian._discount * np.mean(pw_asian_delta)
    print(f"  Pathwise Delta: {asian_delta:.6f}")

    # ------------------------------------------------------------------
    # Barrier Option
    # ------------------------------------------------------------------
    barrier = BarrierOption(K, B, T, r)

    with Timer("Barrier Option Pricing"):
        barrier_price, knockout_prob = barrier.price_with_knockout_info(paths)

    barrier_payoffs = barrier._discount * barrier.payoff(paths)
    barrier_mean, barrier_lo, barrier_hi = confidence_interval(barrier_payoffs)

    print(f"\nBarrier Call (Up-and-Out, B={B}, N={N:,}, steps={N_steps}):")
    print(f"  Price: {barrier_price:.6f} [{barrier_lo:.6f}, {barrier_hi:.6f}] 95% CI")
    print(f"  Knockout Probability: {knockout_prob:.4%}")

    # Likelihood Ratio Delta for Barrier (Pathwise would FAIL here!)
    lr_scores = LikelihoodRatioGreeks.score_delta_gbm(
        paths[:, -1], S0, sigma, T, r
    )
    barrier_lr_delta = LikelihoodRatioGreeks.delta(
        barrier.payoff(paths), lr_scores, barrier._discount
    )
    print(f"  Likelihood Ratio Delta: {barrier_lr_delta:.6f}")
    print(f"  (Pathwise Delta would be INCORRECT for barrier options)")

    return asian_price, barrier_price


# ============================================================================
# SECTION 3: Heston Model — All Option Types
# ============================================================================

def run_heston() -> None:
    """
    Price all three option types under the Heston stochastic volatility model.

    This is the most computationally intensive section. Euler-Maruyama
    requires small time steps for accuracy.
    """
    print("\n" + "=" * 72)
    print("SECTION 3: Heston Stochastic Volatility Model")
    print("=" * 72)

    S0 = config["S0"]
    K = config["K"]
    T = config["T"]
    r = config["r"]
    B = config["B"]
    v0 = config["v0"]
    kappa = config["kappa"]
    theta = config["theta"]
    xi = config["xi"]
    rho = config["rho"]
    N = config["N_paths_pricing"]
    N_steps = config["N_steps_pricing"]

    set_seed(config["base_seed"])

    heston = HestonPath(S0, v0, r, kappa, theta, xi, rho)

    with Timer("Heston Full Path Simulation"):
        S_paths, v_paths = heston.simulate_paths(T, N, N_steps)

    # ------------------------------------------------------------------
    # European under Heston
    # ------------------------------------------------------------------
    european = EuropeanOption(K, T, r)

    with Timer("Heston European Pricing"):
        heston_eu_price = european.price_from_terminal(S_paths[:, -1])

    eu_payoffs = european._discount * european.payoff(S_paths[:, -1])
    eu_mean, eu_lo, eu_hi = confidence_interval(eu_payoffs)

    print(f"\nEuropean Call (Heston, N={N:,}, steps={N_steps}):")
    print(f"  Price: {heston_eu_price:.6f} [{eu_lo:.6f}, {eu_hi:.6f}] 95% CI")
    print(f"  Avg Terminal Variance: {np.mean(v_paths[:, -1]):.6f}")

    # ------------------------------------------------------------------
    # Asian under Heston
    # ------------------------------------------------------------------
    asian = AsianOption(K, T, r)

    with Timer("Heston Asian Pricing"):
        heston_asian_price = asian.price(S_paths)

    asian_payoffs = asian._discount * asian.payoff(S_paths)
    asian_mean, asian_lo, asian_hi = confidence_interval(asian_payoffs)

    print(f"\nAsian Call (Heston, N={N:,}, steps={N_steps}):")
    print(f"  Price: {heston_asian_price:.6f} [{asian_lo:.6f}, {asian_hi:.6f}] 95% CI")

    # ------------------------------------------------------------------
    # Barrier under Heston
    # ------------------------------------------------------------------
    barrier = BarrierOption(K, B, T, r)

    with Timer("Heston Barrier Pricing"):
        heston_barrier_price, knockout_prob = barrier.price_with_knockout_info(S_paths)

    barrier_payoffs = barrier._discount * barrier.payoff(S_paths)
    barrier_mean, barrier_lo, barrier_hi = confidence_interval(barrier_payoffs)

    print(f"\nBarrier Call (Heston, B={B}, N={N:,}, steps={N_steps}):")
    print(f"  Price: {heston_barrier_price:.6f} [{barrier_lo:.6f}, {barrier_hi:.6f}] 95% CI")
    print(f"  Knockout Probability: {knockout_prob:.4%}")

    return heston_eu_price, heston_asian_price, heston_barrier_price


# ============================================================================
# SECTION 4: Convergence Analysis
# ============================================================================

def run_convergence_analysis() -> None:
    """
    Generate convergence data and plots.

    1. Error vs Number of Paths (N):
       For a fixed dt, vary N and show error ∝ 1/sqrt(N).

    2. Bias vs Time Step (dt):
       For a fixed large N (so MC error is negligible), vary dt
       and show the Euler-Maruyama discretization bias ∝ dt.
    """
    print("\n" + "=" * 72)
    print("SECTION 4: Convergence Analysis")
    print("=" * 72)

    S0 = config["S0"]
    K = config["K"]
    T = config["T"]
    r = config["r"]
    sigma = config["sigma"]
    v0 = config["v0"]
    kappa = config["kappa"]
    theta = config["theta"]
    xi = config["xi"]
    rho = config["rho"]

    bs_price = black_scholes_price(S0, K, T, r, sigma)

    # ------------------------------------------------------------------
    # 4.1: Convergence with N (Black-Scholes European)
    # ------------------------------------------------------------------
    print("\n--- 4.1: Error vs Number of Paths (Black-Scholes) ---")

    N_values = config["N_paths_convergence"]
    errors = []
    prices = []
    std_errors = []

    for N in N_values:
        set_seed(config["base_seed"])  # Same seed for each N
        gbm = GBMPath(S0, r, sigma)
        option = EuropeanOption(K, T, r)
        S_T = gbm.simulate_terminal(T, N)
        discounted = option._discount * option.payoff(S_T)

        mc_price = np.mean(discounted)
        std_err = np.std(discounted, ddof=1) / np.sqrt(N)
        err = abs(mc_price - bs_price)

        prices.append(mc_price)
        errors.append(err)
        std_errors.append(std_err)
        print(f"  N={N:>7,}: price={mc_price:.6f}, error={err:.8f}, std_err={std_err:.8f}")

    # Save convergence data
    save_results_csv(
        "convergence_N.csv",
        ["N", "price", "error", "std_error", "true_price"],
        [[n, p, e, s, bs_price] for n, p, e, s in zip(N_values, prices, errors, std_errors)],
    )

    # Plot
    plot_convergence_N(
        N_values=N_values,
        errors=errors,
        true_price=bs_price,
        mc_prices=prices,
        std_errors=std_errors,
        option_type="European Call (Black-Scholes)",
        save=True,
    )

    # ------------------------------------------------------------------
    # 4.2: Discretization Bias (Heston, European)
    # ------------------------------------------------------------------
    print("\n--- 4.2: Discretization Bias vs Time Step (Heston) ---")

    N_large = 200_000  # Large enough that MC error is small
    step_values = config["N_steps_discretization"]
    dt_values = [T / steps for steps in step_values]
    heston_prices = []

    # Reference price with very fine discretization
    set_seed(config["base_seed"])
    heston_fine = HestonPath(S0, v0, r, kappa, theta, xi, rho)
    S_fine, _ = heston_fine.simulate_paths(T, N_large, 504)  # Twice-daily
    option_ref = EuropeanOption(K, T, r)
    ref_price = option_ref.price_from_terminal(S_fine[:, -1])

    print(f"  Reference price (steps=504, N={N_large:,}): {ref_price:.6f}")

    for steps in step_values:
        set_seed(config["base_seed"])
        heston = HestonPath(S0, v0, r, kappa, theta, xi, rho)
        S_paths, _ = heston.simulate_paths(T, N_large, steps)
        option = EuropeanOption(K, T, r)
        price = option.price_from_terminal(S_paths[:, -1])
        heston_prices.append(price)
        bias = price - ref_price
        dt = T / steps
        print(f"  steps={steps:>4}, dt={dt:.6f}: price={price:.6f}, bias={bias:.8f}")

    # Save discretization data
    save_results_csv(
        "convergence_dt.csv",
        ["steps", "dt", "price", "bias", "ref_price"],
        [[s, T/s, p, p - ref_price, ref_price]
         for s, p in zip(step_values, heston_prices)],
    )

    # Plot
    plot_discretization_bias(
        dt_values=dt_values,
        prices=heston_prices,
        true_price=ref_price,
        option_type="European Call (Heston)",
        save=True,
    )

    return N_values, errors, dt_values, heston_prices


# ============================================================================
# SECTION 5: Greeks Convergence Study
# ============================================================================

def run_greeks_convergence() -> None:
    """
    Study the convergence of pathwise vs likelihood ratio for Delta.

    Demonstrates:
    - Both converge at 1/sqrt(N).
    - Pathwise has lower variance (tighter error bars).
    - Pathwise FAILS for barrier options (not shown here; see README).
    """
    print("\n" + "=" * 72)
    print("SECTION 5: Greeks Convergence Analysis")
    print("=" * 72)

    S0 = config["S0"]
    K = config["K"]
    T = config["T"]
    r = config["r"]
    sigma = config["sigma"]

    bs_delta = black_scholes_delta(S0, K, T, r, sigma)
    N_values = config["N_paths_convergence"]

    pw_errors = []
    lr_errors = []
    pw_variances = []
    lr_variances = []

    for N in N_values:
        set_seed(config["base_seed"])
        gbm = GBMPath(S0, r, sigma)
        option = EuropeanOption(K, T, r)
        S_T = gbm.simulate_terminal(T, N)
        payoffs = option.payoff(S_T)

        # Pathwise Delta
        pw_delta_path = PathwiseGreeks.delta_european(S_T, S0, K, T, r)
        pw_estimates = option._discount * pw_delta_path
        pw_delta = np.mean(pw_estimates)
        pw_var = np.var(pw_estimates, ddof=1)

        # Likelihood Ratio Delta
        lr_scores = LikelihoodRatioGreeks.score_delta_gbm(S_T, S0, sigma, T, r)
        lr_estimates = option._discount * payoffs * lr_scores
        lr_delta = np.mean(lr_estimates)
        lr_var = np.var(lr_estimates, ddof=1)

        pw_err = abs(pw_delta - bs_delta)
        lr_err = abs(lr_delta - bs_delta)

        pw_errors.append(pw_err)
        lr_errors.append(lr_err)
        pw_variances.append(pw_var)
        lr_variances.append(lr_var)

        print(f"  N={N:>7,}: PW Delta={pw_delta:.6f} (err={pw_err:.8f}, var={pw_var:.8f})")
        print(f"           LR Delta={lr_delta:.6f} (err={lr_err:.8f}, var={lr_var:.8f})")

    # Save Greeks convergence data
    save_results_csv(
        "greeks_convergence.csv",
        ["N", "pw_delta_error", "lr_delta_error", "pw_variance", "lr_variance", "true_delta"],
        [[n, pe, le, pv, lv, bs_delta]
         for n, pe, le, pv, lv in zip(N_values, pw_errors, lr_errors, pw_variances, lr_variances)],
    )

    # Plot error convergence
    plot_greeks_convergence(
        N_values=N_values,
        pathwise_errors=pw_errors,
        lr_errors=lr_errors,
        greek_name="Delta",
        save=True,
    )

    # Plot variance comparison
    plot_variance_comparison(
        N_values=N_values,
        pathwise_var=pw_variances,
        lr_var=lr_variances,
        greek_name="Delta",
        save=True,
    )

    return pw_errors, lr_errors


# ============================================================================
# SECTION 6: Greeks Comparison Plot
# ============================================================================

def run_greeks_comparison_plot(delta_results: dict, vega_results: dict) -> None:
    """
    Generate bar charts comparing all Greek methods against the true value.
    """
    print("\n" + "=" * 72)
    print("SECTION 6: Greeks Comparison Plots")
    print("=" * 72)

    bs_delta = delta_results.get("black_scholes", 0)
    bs_vega = vega_results.get("black_scholes", 0)

    plot_greeks_comparison(
        greek_name="Delta",
        pathwise_value=delta_results.get("pathwise", 0),
        lr_value=delta_results.get("likelihood_ratio", 0),
        fd_value=delta_results.get("finite_difference", 0),
        true_value=bs_delta,
        save=True,
    )

    plot_greeks_comparison(
        greek_name="Vega",
        pathwise_value=vega_results.get("pathwise", 0),
        lr_value=vega_results.get("likelihood_ratio", 0),
        fd_value=vega_results.get("finite_difference", 0),
        true_value=bs_vega,
        save=True,
    )

    print("  Delta and Vega comparison plots saved to ../plots/")


# ============================================================================
# MAIN — Run everything
# ============================================================================

def main() -> None:
    """
    Execute the full Monte Carlo option pricing pipeline.

    Order matters: we run Black-Scholes validation first to confirm
    the engine is correct before moving to more complex models.
    """
    print("=" * 72)
    print("MONTE CARLO OPTION PRICER")
    print("Author: Claudia Maria Lopez Bombin")
    print("Models: Black-Scholes + Heston Stochastic Volatility")
    print("Options: European, Asian (Arithmetic), Barrier (Up-and-Out)")
    print("Greeks: Pathwise, Likelihood Ratio, Finite Difference")
    print("=" * 72)

    # Create output directories
    ensure_dir("../data")
    ensure_dir("../plots")

    # ------------------------------------------------------------------
    # Run all sections
    # ------------------------------------------------------------------
    with Timer("Total Pipeline Runtime"):

        # Section 1: Black-Scholes validation
        delta_results, vega_results = run_black_scholes_validation()

        # Section 2: Path-dependent options
        run_path_dependent_options()

        # Section 3: Heston model
        run_heston()

        # Section 4: Convergence analysis
        run_convergence_analysis()

        # Section 5: Greeks convergence
        run_greeks_convergence()

        # Section 6: Greeks comparison plots
        run_greeks_comparison_plot(delta_results, vega_results)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    print("\nOutput files:")
    print("  ../data/convergence_N.csv      — Error vs number of paths")
    print("  ../data/convergence_dt.csv     — Discretization bias vs dt")
    print("  ../data/greeks_convergence.csv — Greeks convergence data")
    print("  ../plots/convergence_N.png     — Convergence plot")
    print("  ../plots/convergence_dt.png    — Discretization bias plot")
    print("  ../plots/delta_comparison.png  — Delta method comparison")
    print("  ../plots/vega_comparison.png   — Vega method comparison")
    print("  ../plots/delta_convergence.png — Delta convergence")
    print("  ../plots/delta_variance.png    — Variance comparison")
    print()


if __name__ == "__main__":
    main()