"""
Stochastic Process Simulation Engines
======================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/monte-carlo-option-pricer

This module implements from-scratch Monte Carlo path generation for
the two models required by the project:

1. Black-Scholes Model (Geometric Brownian Motion)
   The "Hello World" of quantitative finance. Serves as the baseline
   to validate that the Monte Carlo engine is working correctly by
   comparing against the closed-form Black-Scholes formula.

   SDE:  dS_t = r * S_t * dt + sigma * S_t * dW_t

   Exact solution (no discretization error):
   S_t = S_0 * exp( (r - sigma^2/2) * t + sigma * W_t )

2. Heston Model (Stochastic Volatility)
   A more realistic model where volatility is not constant but follows
   a mean-reverting CIR process. This demonstrates that the engine goes
   beyond the basics and handles correlated Brownian motions.

   SDE system:
   dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW1_t
   dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW2_t
   <dW1_t, dW2_t> = rho * dt

DESIGN PRINCIPLES
-----------------
- Vectorized over paths: All simulations operate on (N_paths, N_steps)
  arrays. No Python for-loop over paths. This is the single most
  important performance decision — it makes 1M paths feasible in
  pure Python/NumPy.
- Minimal dependencies: Only NumPy. No QuantLib, no TensorFlow.
- Reproducibility: A module-level RNG with set_seed() allows exact
  reproduction of any simulation run.
- Full paths stored: Needed for path-dependent options (Asian,
  barrier), not just terminal values.

References
----------
Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
    Springer. Chapters 3 and 4.
Heston, S. (1993). A Closed-Form Solution for Options with Stochastic
    Volatility with Applications to Bond and Currency Options.
    The Review of Financial Studies, 6(2), 327-343.
"""

import numpy as np
from typing import Tuple, Optional


# ============================================================================
# GLOBAL RANDOM NUMBER GENERATOR
# ============================================================================
# We expose a module-level generator so the user can fix the seed once
# and have it propagate to all subsequent simulations. This is critical
# for reproducibility and for comparing Greeks methods on identical paths.
#
# NumPy's default_rng() uses the PCG-64 bit generator, which is
# statistically superior to the legacy MT19937 and has better
# performance characteristics for large-scale simulations.
# ============================================================================

_rng = np.random.default_rng()


def set_seed(seed: int) -> None:
    """
    Set the global random seed for reproducible simulations.

    All subsequent calls to GBMPath.simulate_* or HestonPath.simulate_paths
    will use this seeded generator unless a local seed is provided.

    Parameters
    ----------
    seed : int
        Seed for NumPy's default PCG-64 bit generator.
        Example: set_seed(42)
    """
    global _rng
    _rng = np.random.default_rng(seed)


# ============================================================================
# BLACK-SCHOLES MODEL — Geometric Brownian Motion
# ============================================================================
# This is the baseline model. We implement it first to validate the Monte
# Carlo engine against the closed-form Black-Scholes formula.
#
# KEY INSIGHT: GBM has an exact solution. We don't need Euler-Maruyama
# discretization. We can jump directly to any time t:
#
#   S_t = S_0 * exp( (r - sigma^2/2) * t + sigma * W_t )
#
# where W_t ~ N(0, t). This means:
# - Terminal simulation: O(1) per path, no discretization error.
# - Path simulation: We still compute the exact S_t at each monitoring
#   date; the Brownian path W_t is built from independent increments.
#
# The drift term (r - sigma^2/2) is the risk-neutral drift. Under the
# risk-neutral measure, the asset grows at the risk-free rate r, but
# the geometric average is reduced by sigma^2/2 due to Ito's lemma.
# ============================================================================

class GBMPath:
    """
    Geometric Brownian Motion path generator.

    Simulates asset price paths under the Black-Scholes model using
    the exact solution to the SDE. No time discretization is needed
    for GBM — we can jump directly to any monitoring date.

    S_t = S_0 * exp( (r - 0.5 * sigma^2) * t + sigma * W_t )

    Attributes
    ----------
    S0 : float
        Initial asset price (e.g., 100.0).
    r : float
        Risk-free interest rate, continuously compounded (e.g., 0.05 for 5%).
    sigma : float
        Constant volatility (e.g., 0.20 for 20% annualized).
    """

    def __init__(self, S0: float, r: float, sigma: float) -> None:
        """
        Initialize the GBM model.

        Parameters
        ----------
        S0 : float
            Initial asset price. Must be > 0.
        r : float
            Risk-free rate. Can be negative in low-rate environments.
        sigma : float
            Volatility. Must be > 0.

        Raises
        ------
        ValueError
            If S0 <= 0 or sigma <= 0.
        """
        if S0 <= 0:
            raise ValueError(f"S0 must be positive, got {S0}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.S0 = S0
        self.r = r
        self.sigma = sigma

    def simulate_terminal(
        self, T: float, N_paths: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate asset price at maturity T for N_paths independent paths.

        Uses the exact solution — ZERO discretization error. Each path
        is generated in O(1) as:

            S_T = S_0 * exp( (r - sigma^2/2) * T + sigma * sqrt(T) * Z )

        where Z ~ N(0, 1).

        Parameters
        ----------
        T : float
            Time to maturity in years (e.g., 1.0 for 1 year).
        N_paths : int
            Number of Monte Carlo paths.
        seed : int, optional
            Local seed for this specific call. Does not affect the global RNG.
            Useful for generating independent batches.

        Returns
        -------
        S_T : np.ndarray of shape (N_paths,)
            Terminal asset prices at time T.
        """
        # Use local RNG if seed provided, otherwise global
        rng = _rng if seed is None else np.random.default_rng(seed)

        # Standard normal random variables — one per path
        Z = rng.normal(0.0, 1.0, size=N_paths)

        # Exact GBM solution
        drift = (self.r - 0.5 * self.sigma ** 2) * T
        diffusion = self.sigma * np.sqrt(T) * Z

        return self.S0 * np.exp(drift + diffusion)

    def simulate_paths(
        self,
        T: float,
        N_paths: int,
        N_steps: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate full price paths with N_steps equally spaced monitoring dates.

        Generates a (N_paths, N_steps + 1) array where column 0 is S_0
        and column k is S_{t_k} with t_k = k * dt.

        While GBM has an exact solution, full paths are required for
        path-dependent options:
        - Asian options need the arithmetic average along the path.
        - Barrier options need to monitor if the asset ever crosses a level.

        The simulation uses the exact solution on a time grid. The Brownian
        motion W_t is constructed from independent increments:
            W_{t_k} = sum_{i=1}^{k} sqrt(dt) * Z_i
        where Z_i ~ N(0, 1) are independent standard normals.

        Parameters
        ----------
        T : float
            Time to maturity in years.
        N_paths : int
            Number of independent paths to generate.
        N_steps : int
            Number of time steps per path.
        seed : int, optional
            Local seed for this call.

        Returns
        -------
        paths : np.ndarray of shape (N_paths, N_steps + 1)
            Asset price at each monitoring date. paths[:, 0] = S0.
        """
        rng = _rng if seed is None else np.random.default_rng(seed)
        dt = T / N_steps

        # ------------------------------------------------------------------
        # Step 1: Generate Brownian increments
        # dW ~ N(0, sqrt(dt)) independently for each path and each step
        # Shape: (N_paths, N_steps)
        # ------------------------------------------------------------------
        dW = rng.normal(0.0, np.sqrt(dt), size=(N_paths, N_steps))

        # ------------------------------------------------------------------
        # Step 2: Cumulative sum gives W_t at each monitoring date
        # W[:, k] = sum_{i=0}^{k-1} dW[:, i]  (Brownian path)
        # Prepend W_0 = 0 as the first column.
        # Shape: (N_paths, N_steps + 1)
        # ------------------------------------------------------------------
        W = np.cumsum(dW, axis=1)
        W = np.concatenate(
            [np.zeros((N_paths, 1)), W], axis=1
        )

        # ------------------------------------------------------------------
        # Step 3: Time grid for drift computation
        # t = [0, dt, 2*dt, ..., T]
        # Shape: (N_steps + 1,)
        # ------------------------------------------------------------------
        t = np.linspace(0, T, N_steps + 1)

        # ------------------------------------------------------------------
        # Step 4: Exact GBM solution on the time grid
        # S_t = S_0 * exp( (r - sigma^2/2) * t + sigma * W_t )
        # Broadcasting: t is (N_steps+1,), W is (N_paths, N_steps+1)
        # Result: (N_paths, N_steps+1)
        # ------------------------------------------------------------------
        drift = (self.r - 0.5 * self.sigma ** 2) * t
        diffusion = self.sigma * W

        return self.S0 * np.exp(drift + diffusion)


# ============================================================================
# HESTON MODEL — Stochastic Volatility with Mean-Reverting Variance
# ============================================================================
# The Heston model (1993) is the workhorse of stochastic volatility modeling.
# Unlike Black-Scholes where sigma is constant, Heston models variance v_t
# as a mean-reverting CIR process:
#
#   dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW2_t
#
# Key parameters:
# - kappa: Speed of mean reversion. Higher = faster pull to theta.
# - theta: Long-run mean variance. Volatility tends to sqrt(theta).
# - xi:    Volatility of variance ("vol of vol"). Controls how noisy v_t is.
# - rho:   Correlation between asset and variance shocks.
#          rho < 0 captures the "leverage effect": when stock prices drop,
#          volatility tends to rise (negative skew in returns).
#
# Unlike GBM, the Heston SDE system has NO closed-form solution for S_t.
# We must use discretization. Euler-Maruyama is the simplest scheme.
#
# VARIANCE BOUNDARY HANDLING
# ---------------------------
# The CIR process theoretically stays positive if the Feller condition holds:
#     2 * kappa * theta >= xi^2
# When this condition is violated, the Euler discretization can produce
# negative variance values.
#
# We handle this with ABSORPTION (truncation to zero):
#     v_t = max(v_t, 0)
# This is the simplest and most common fix in industry practice. It biases
# the variance slightly upward (we replace negatives with zero instead of
# letting them become positive through reflection), but the bias diminishes
# as dt -> 0.
#
# Alternatives not implemented:
# - Reflection: v_t = |v_t| (less bias but changes the distribution)
# - Alfonsi (2005) implicit scheme: more accurate but slower
# - Quadratic-exponential (QE) scheme: popular in practice for large dt
#
# We document this choice because it's a known source of discretization
# bias that appears in the convergence plots (bias vs dt).
# ============================================================================

class HestonPath:
    """
    Heston stochastic volatility path generator.

    Simulates asset price AND variance paths under the Heston (1993)
    model using Euler-Maruyama discretization.

    SDE system:
        dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW1_t
        dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW2_t
        <dW1_t, dW2_t> = rho * dt

    The correlation rho between the two Brownian motions is achieved
    via Cholesky decomposition of the covariance matrix.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    v0 : float
        Initial variance (NOT volatility). v0 = sigma_0^2.
        Example: sigma_0 = 0.20 => v0 = 0.04.
    r : float
        Risk-free interest rate, continuously compounded.
    kappa : float
        Mean-reversion speed of variance (> 0).
        Higher values = faster reversion to theta.
    theta : float
        Long-run mean of variance (> 0).
        Long-run volatility = sqrt(theta).
    xi : float
        Volatility of variance, "vol of vol" (> 0).
        Controls the noise in the variance process.
    rho : float
        Correlation between asset and variance Brownian motions.
        Must be in [-1, 1]. Typically negative (-0.7 to -0.3)
        to capture the leverage effect.
    """

    def __init__(
        self,
        S0: float,
        v0: float,
        r: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
    ) -> None:
        """
        Initialize the Heston model.

        Parameters
        ----------
        S0 : float
            Initial asset price. Must be > 0.
        v0 : float
            Initial variance. Must be > 0.
        r : float
            Risk-free rate.
        kappa : float
            Mean-reversion speed. Must be > 0.
        theta : float
            Long-run variance mean. Must be > 0.
        xi : float
            Volatility of variance. Must be > 0.
        rho : float
            Correlation, must be in [-1, 1].

        Raises
        ------
        ValueError
            If any parameter is outside its valid range.
        """
        # ------------------------------------------------------------------
        # Input validation — fail fast with clear messages
        # ------------------------------------------------------------------
        if S0 <= 0:
            raise ValueError(f"S0 must be positive, got {S0}")
        if v0 <= 0:
            raise ValueError(f"v0 (initial variance) must be positive, got {v0}")
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        if xi <= 0:
            raise ValueError(f"xi (vol of vol) must be positive, got {xi}")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {rho}")

        self.S0 = S0
        self.v0 = v0
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    def simulate_paths(
        self,
        T: float,
        N_paths: int,
        N_steps: int,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston paths via Euler-Maruyama discretization.

        Generates correlated asset price AND variance paths.

        EULER-MARUYAMA SCHEME (for each path, at each step k):

            v_{k+1} = v_k + kappa * (theta - v_k) * dt
                           + xi * sqrt(max(v_k, 0)) * sqrt(dt) * Z2_k
            v_{k+1} = max(v_{k+1}, 0)     # Absorption at zero

            S_{k+1} = S_k + r * S_k * dt
                           + sqrt(max(v_k, 0)) * S_k * sqrt(dt) * Z1_k

        where (Z1_k, Z2_k) are correlated standard normals:
            Z1_k = e1_k
            Z2_k = rho * e1_k + sqrt(1 - rho^2) * e2_k
        with e1_k, e2_k ~ N(0, 1) independent.

        Parameters
        ----------
        T : float
            Time to maturity in years.
        N_paths : int
            Number of independent paths.
        N_steps : int
            Number of time steps per path.
        seed : int, optional
            Local seed for this call.

        Returns
        -------
        S : np.ndarray of shape (N_paths, N_steps + 1)
            Asset price paths. S[:, 0] = S0.
        v : np.ndarray of shape (N_paths, N_steps + 1)
            Variance paths. v[:, 0] = v0.
        """
        rng = _rng if seed is None else np.random.default_rng(seed)
        dt = T / N_steps

        # ------------------------------------------------------------------
        # Pre-allocate arrays for asset prices and variance
        # We store full paths because Asian/barrier options need every
        # intermediate monitoring date.
        # Shape: (N_paths, N_steps + 1)
        # ------------------------------------------------------------------
        S = np.zeros((N_paths, N_steps + 1))
        v = np.zeros((N_paths, N_steps + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        # ------------------------------------------------------------------
        # Pre-compute Cholesky factor for the correlation structure
        #
        # Covariance matrix of (dW1, dW2):
        #     [[1,  rho],
        #      [rho, 1 ]]
        #
        # Cholesky decomposition L such that L @ L.T = Cov:
        #     L = [[1,          0],
        #          [rho, sqrt(1 - rho^2)]]
        #
        # Then: [dW1, dW2].T = L @ [e1, e2].T
        #       dW1 = e1
        #       dW2 = rho * e1 + sqrt(1 - rho^2) * e2
        # where e1, e2 ~ N(0, dt) independent.
        # ------------------------------------------------------------------
        rho_complement = np.sqrt(1.0 - self.rho ** 2)

        # ------------------------------------------------------------------
        # Euler-Maruyama time-stepping loop
        #
        # We iterate over time steps (not paths). At each step we generate
        # N_paths independent Gaussian pairs and update all paths in parallel.
        # This is vectorized over paths but sequential over time — the
        # variance path must be built step by step because each step depends
        # on the previous variance value.
        # ------------------------------------------------------------------
        for k in range(N_steps):
            # Current state for all paths
            S_k = S[:, k]
            v_k = v[:, k]

            # Enforce non-negative variance before computing sqrt
            # This is the absorption fix: v_k = max(v_k, 0)
            v_k_pos = np.maximum(v_k, 0.0)

            # Generate independent standard normals for this step
            e1 = rng.normal(0.0, 1.0, size=N_paths)
            e2 = rng.normal(0.0, 1.0, size=N_paths)

            # Correlate the Brownian increments via Cholesky
            # dW1 ~ N(0, dt), dW2 ~ N(0, dt), Corr(dW1, dW2) = rho
            dW1 = np.sqrt(dt) * e1
            dW2 = np.sqrt(dt) * (self.rho * e1 + rho_complement * e2)

            # --------------------------------------------------------------
            # Update variance: CIR process
            # v_{k+1} = v_k + kappa*(theta - v_k)*dt + xi*sqrt(v_k)*dW2
            # --------------------------------------------------------------
            v_new = (
                v_k
                + self.kappa * (self.theta - v_k) * dt
                + self.xi * np.sqrt(v_k_pos) * dW2
            )

            # Absorption at zero: truncate negative variance
            # This is the most common practical fix for Euler-discretized
            # CIR processes that violate the Feller condition.
            v_new = np.maximum(v_new, 0.0)

            # --------------------------------------------------------------
            # Update asset price: GBM with stochastic volatility
            # S_{k+1} = S_k + r*S_k*dt + sqrt(v_k)*S_k*dW1
            #
            # NOTE: We use the Euler scheme for S as well (not the exact
            # GBM solution) because v_t is not constant along the step.
            # Using the exact GBM solution with average v would be a
            # possible improvement but is not standard practice.
            # --------------------------------------------------------------
            S_new = S_k + self.r * S_k * dt + np.sqrt(v_k_pos) * S_k * dW1

            # Store updated values
            S[:, k + 1] = S_new
            v[:, k + 1] = v_new

        return S, v