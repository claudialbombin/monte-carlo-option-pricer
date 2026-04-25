"""
Greeks Computation — Pathwise and Likelihood Ratio Methods
============================================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/monte-carlo-option-pricer

This module implements two fundamentally different approaches to
computing option price sensitivities (Greeks) via Monte Carlo simulation.

WHAT ARE GREEKS?
-----------------
Greeks measure how an option's price changes when a model parameter changes.
They are essential for:
- Hedging: How many shares to hold to offset option risk (Delta).
- Risk management: How sensitive is the portfolio to volatility moves (Vega).
- Trading: Identifying mispriced options relative to the model.

In Jane Street and other trading firms, Greeks are computed continuously
to manage the firm's exposure across thousands of positions.

TWO METHODS IMPLEMENTED
-----------------------

1. PATHWISE SENSITIVITIES (Infinitesimal Perturbation Analysis)
   - Derive the payoff function with respect to the parameter.
   - Compute the expected value of that derivative along each path.
   - PRO: Low variance. Uses the same paths as the price.
   - CON: Requires payoff to be differentiable (Lipschitz continuous).
     FAILS for discontinuous payoffs (barrier options, digital options).
   
   Formula: Greek = E[ d(payoff) / d(parameter) ]
   
   Example — Delta (sensitivity to S0):
   European call: payoff = max(S_T - K, 0)
   d(payoff)/dS0 = (d(payoff)/dS_T) * (dS_T/dS0)
                  = indicator(S_T > K) * (S_T / S0)
   because S_T = S0 * exp(...) => dS_T/dS0 = S_T/S0.

2. LIKELIHOOD RATIO METHOD (Score Function Method)
   - Differentiate the probability density of S_T, not the payoff.
   - Weight each payoff by a "score function" (derivative of log-density).
   - PRO: Works for ANY payoff — even discontinuous barrier options.
     This is the key advantage over pathwise.
   - CON: Higher variance. The score function can take extreme values
     in the tails, increasing the standard error.
   
   Formula: Greek = E[ payoff * d(log density) / d(parameter) ]
   
   The score function for Delta under GBM:
   d(log p(S_T | S0, sigma, r, T)) / d(S0) = ???

   Wait — we differentiate the transition density from S0 to S_T.
   For GBM, log(S_T / S0) ~ N((r - sigma^2/2)*T, sigma^2 * T).
   The score for S0 is: (log(S_T/S0) - (r - sigma^2/2)*T) / (S0 * sigma^2 * T)

WHY TWO METHODS?
-----------------
This is the central demonstration of the project:
- For European and Asian options (continuous payoffs), pathwise gives
  lower variance than likelihood ratio.
- For barrier options (discontinuous payoffs), pathwise FAILS COMPLETELY
  (gives wrong answer) because the payoff has a jump at the barrier.
  Likelihood ratio works because it doesn't touch the payoff function.

We validate both methods against:
- Finite differences (perturb the parameter, reprice, take difference).
- Closed-form Black-Scholes Greeks (for European options only).

References
----------
Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
    Springer. Chapter 7: "Estimating Sensitivities" — THE key reference.
Broadie, M. & Glasserman, P. (1996). Estimating Security Price Derivatives
    Using Simulation. Management Science, 42(2), 269-285.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy.stats import norm


# ============================================================================
# FINITE DIFFERENCE — The baseline validation method
# ============================================================================
# Finite difference is the simplest Greek computation:
#   Greek ≈ (Price(param + h) - Price(param - h)) / (2h)
#
# It requires REPRICING the option twice with perturbed parameters.
# This is expensive (2x simulation cost) but works for ANY option type
# and serves as our ground truth for validating pathwise and likelihood
# ratio methods.
#
# We use CENTRAL differences (h and -h) because they have O(h^2) bias,
# superior to forward/backward differences which have O(h) bias.
# ============================================================================

def finite_difference_delta(
    price_func: Callable[[float], float],
    S0: float,
    h: float = 1e-4,
) -> float:
    """
    Compute Delta via central finite difference.

    Delta = d(Price) / d(S0) ≈ [Price(S0 + h) - Price(S0 - h)] / (2h)

    Parameters
    ----------
    price_func : callable
        Function that takes S0 and returns the option price.
        Must be deterministic (same random seed for both calls).
    S0 : float
        Current asset price.
    h : float
        Perturbation size. Default 1e-4 works well for most cases.

    Returns
    -------
    delta : float
        Finite difference estimate of Delta.
    """
    price_up = price_func(S0 + h)
    price_down = price_func(S0 - h)
    return (price_up - price_down) / (2.0 * h)


def finite_difference_vega(
    price_func: Callable[[float], float],
    sigma: float,
    h: float = 1e-4,
) -> float:
    """
    Compute Vega via central finite difference.

    Vega = d(Price) / d(sigma) ≈ [Price(sigma + h) - Price(sigma - h)] / (2h)

    NOTE: Vega is conventionally quoted as the change in price per
    1 PERCENTAGE POINT change in volatility (e.g., from 20% to 21%).
    Some conventions divide by 100. We do NOT divide by 100 here.

    Parameters
    ----------
    price_func : callable
        Function that takes sigma and returns the option price.
    sigma : float
        Current volatility (as decimal, e.g., 0.20 for 20%).
    h : float
        Perturbation size.

    Returns
    -------
    vega : float
        Finite difference estimate of Vega.
    """
    price_up = price_func(sigma + h)
    price_down = price_func(sigma - h)
    return (price_up - price_down) / (2.0 * h)


# ============================================================================
# CLOSED-FORM BLACK-SCHOLES GREEKS — Ground truth for European options
# ============================================================================
# For European call options under Black-Scholes, we have analytical formulas
# for both price and Greeks. These serve as the ground truth for validating
# Monte Carlo estimates.
#
# The formulas below are the standard ones from any derivatives textbook.
# We use scipy.stats.norm for the standard normal CDF (N) and PDF (N').
#
# Derivation: Differentiate the Black-Scholes formula
#   C = S_0 * N(d1) - K * e^{-rT} * N(d2)
# with respect to each parameter.
# ============================================================================

def black_scholes_price(
    S0: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Black-Scholes European call option price (closed form).

    Parameters
    ----------
    S0 : float
        Initial asset price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.

    Returns
    -------
    price : float
        Black-Scholes call price.
    """
    if T <= 0:
        # At expiry: intrinsic value
        return max(S0 - K, 0.0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_delta(
    S0: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Black-Scholes Delta for a European call.

    Delta = dC/dS0 = N(d1)

    For a call, Delta is always in [0, 1].
    Deep in-the-money calls → Delta ≈ 1.
    Deep out-of-the-money calls → Delta ≈ 0.
    At-the-money calls → Delta ≈ 0.5.
    """
    if T <= 0:
        return 1.0 if S0 > K else 0.0

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def black_scholes_vega(
    S0: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Black-Scholes Vega for a European call.

    Vega = dC/d(sigma) = S0 * sqrt(T) * N'(d1)

    Note: Vega is the same for calls and puts.
    It's always positive: higher volatility → higher option price.
    Vega is highest for at-the-money options and decays as the
    option moves in or out of the money.

    N'(d1) = (1 / sqrt(2*pi)) * exp(-d1^2 / 2) is the standard normal PDF.
    """
    if T <= 0:
        return 0.0  # Volatility doesn't matter at expiry

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S0 * np.sqrt(T) * norm.pdf(d1)


# ============================================================================
# PATHWISE SENSITIVITIES — Differentiate the payoff function
# ============================================================================
# The pathwise method (also called Infinitesimal Perturbation Analysis, IPA)
# computes Greeks by differentiating the payoff function with respect to the
# parameter of interest and averaging over paths.
#
# REQUIREMENT: The payoff function must be differentiable almost everywhere
# with respect to the parameter AND the derivative must be bounded.
# This holds for European and Asian options but FAILS for barrier options
# because the knockout creates a discontinuity.
#
# WHY IT WORKS (when it works):
# We interchange differentiation and expectation:
#   d/dθ E[payoff(S(θ))] = E[d/dθ payoff(S(θ))]
#   = E[payoff'(S(θ)) * dS/dθ]
#
# The pathwise derivative is computed analytically and evaluated along
# each simulated path. The average over paths gives the Greek.
#
# ADVANTAGE: Same paths as pricing. Low variance. No additional simulation.
# DISADVANTAGE: Requires differentiable payoff. FAILS for barriers.
# ============================================================================

class PathwiseGreeks:
    """
    Pathwise sensitivity estimates for Delta and Vega.

    This class computes Greeks by analytically differentiating the
    payoff function and averaging the pathwise derivatives over
    all simulated paths.

    Works for: EuropeanOption, AsianOption
    Does NOT work for: BarrierOption (payoff is discontinuous at barrier)
    """

    @staticmethod
    def delta_european(
        S_T: np.ndarray,
        S0: float,
        K: float,
        T: float,
        r: float,
    ) -> np.ndarray:
        """
        Pathwise Delta for a European call option.

        Derivation:
        payoff = max(S_T - K, 0)
        d(payoff)/dS0 = d(payoff)/dS_T * dS_T/dS0

        d(payoff)/dS_T = indicator(S_T > K)
        dS_T/dS0 = S_T / S0  (since S_T = S0 * exp(...))

        So: pathwise_delta = indicator(S_T > K) * (S_T / S0)

        Parameters
        ----------
        S_T : np.ndarray of shape (N_paths,)
            Terminal asset prices.
        S0 : float
            Initial asset price.
        K : float
            Strike price.
        T : float
            Time to maturity (unused but kept for API consistency).
        r : float
            Risk-free rate (unused for the pathwise derivative itself,
            but the path generation uses it).

        Returns
        -------
        pathwise_delta : np.ndarray of shape (N_paths,)
            Pathwise Delta contribution for each path.
            Average these and discount to get the Greek.
        """
        # Indicator: is the option in the money?
        in_the_money = (S_T > K).astype(float)

        # Derivative of S_T with respect to S0
        dST_dS0 = S_T / S0

        return in_the_money * dST_dS0

    @staticmethod
    def vega_european(
        S_T: np.ndarray,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> np.ndarray:
        """
        Pathwise Vega for a European call option.

        Derivation:
        S_T = S0 * exp((r - sigma^2/2)*T + sigma*W_T)
        dS_T/d(sigma) = S_T * (W_T - sigma*T)
                      = S_T * (log(S_T/S0) - (r - sigma^2/2)*T - sigma^2*T) / sigma
                      
        Simpler form using the relationship:
        W_T = (log(S_T/S0) - (r - sigma^2/2)*T) / sigma
        dS_T/d(sigma) = S_T * (W_T - sigma*T)

        d(payoff)/d(sigma) = d(payoff)/dS_T * dS_T/d(sigma)
        = indicator(S_T > K) * dS_T/d(sigma)

        Parameters
        ----------
        S_T : np.ndarray of shape (N_paths,)
            Terminal asset prices.
        S0 : float
            Initial asset price.
        K : float
            Strike price.
        T : float
            Time to maturity.
        r : float
            Risk-free rate.
        sigma : float
            Volatility.

        Returns
        -------
        pathwise_vega : np.ndarray of shape (N_paths,)
            Pathwise Vega contribution for each path.
        """
        # Recover the Brownian motion W_T from S_T
        # S_T = S0 * exp((r - sigma^2/2)*T + sigma*W_T)
        # => W_T = (log(S_T/S0) - (r - sigma^2/2)*T) / sigma
        log_return = np.log(S_T / S0)
        drift_component = (r - 0.5 * sigma ** 2) * T
        W_T = (log_return - drift_component) / sigma

        # Derivative of S_T with respect to sigma
        # dS_T/d(sigma) = S_T * (W_T - sigma*T)
        dST_dsigma = S_T * (W_T - sigma * T)

        # Only count in-the-money paths
        in_the_money = (S_T > K).astype(float)

        return in_the_money * dST_dsigma

    @staticmethod
    def delta_asian(
        paths: np.ndarray,
        S0: float,
        K: float,
        T: float,
        r: float,
    ) -> np.ndarray:
        """
        Pathwise Delta for an arithmetic Asian call option.

        Derivation:
        payoff = max(A - K, 0) where A = (1/n) * sum(S_{t_i})
        d(payoff)/dS0 = indicator(A > K) * dA/dS0

        dA/dS0 = (1/n) * sum(dS_{t_i}/dS0)
        dS_{t_i}/dS0 = S_{t_i} / S0

        So: dA/dS0 = (1/n) * sum(S_{t_i} / S0) = A / S0

        Therefore:
        pathwise_delta = indicator(A > K) * (A / S0)

        This is elegantly simple: the pathwise Delta for an Asian option
        is just the average price divided by S0, for paths where the
        average exceeds the strike.

        Parameters
        ----------
        paths : np.ndarray of shape (N_paths, N_steps + 1)
            Full asset price paths.
        S0 : float
            Initial asset price.
        K : float
            Strike price.
        T : float
            Time to maturity (unused, kept for API consistency).
        r : float
            Risk-free rate (unused in pathwise derivative).

        Returns
        -------
        pathwise_delta : np.ndarray of shape (N_paths,)
            Pathwise Delta contribution for each path.
        """
        # Arithmetic average excluding S_0
        A = np.mean(paths[:, 1:], axis=1)

        # Indicator: average > strike
        in_the_money = (A > K).astype(float)

        # dA/dS0 = A / S0
        dA_dS0 = A / S0

        return in_the_money * dA_dS0

    @staticmethod
    def vega_asian(
        paths: np.ndarray,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        N_steps: int,
    ) -> np.ndarray:
        """
        Pathwise Vega for an arithmetic Asian call option.

        Derivation:
        payoff = max(A - K, 0) where A = (1/n) * sum(S_{t_i})
        d(payoff)/d(sigma) = indicator(A > K) * dA/d(sigma)

        dA/d(sigma) = (1/n) * sum(dS_{t_i}/d(sigma))
        dS_{t_i}/d(sigma) = S_{t_i} * (W_{t_i} - sigma*t_i)

        Parameters
        ----------
        paths : np.ndarray of shape (N_paths, N_steps + 1)
            Full asset price paths.
        S0 : float
            Initial asset price.
        K : float
            Strike price.
        T : float
            Time to maturity.
        r : float
            Risk-free rate.
        sigma : float
            Volatility.
        N_steps : int
            Number of time steps used in the path simulation.

        Returns
        -------
        pathwise_vega : np.ndarray of shape (N_paths,)
            Pathwise Vega contribution for each path.
        """
        N_paths = paths.shape[0]
        dt = T / N_steps

        # For each monitoring date t_i, we need W_{t_i}
        # We can recover it from S_{t_i}:
        # W_{t_i} = (log(S_{t_i}/S0) - (r - sigma^2/2)*t_i) / sigma

        t = np.linspace(0, T, N_steps + 1)  # (N_steps+1,)

        # Recover W for all paths and all time steps
        # log(S/S0) shape: (N_paths, N_steps+1)
        log_returns = np.log(paths / S0)
        drift_component = (r - 0.5 * sigma ** 2) * t  # broadcast over paths
        W = (log_returns - drift_component) / sigma

        # dS_{t_i}/d(sigma) = S_{t_i} * (W_{t_i} - sigma * t_i)
        dS_dsigma = paths * (W - sigma * t)  # (N_paths, N_steps+1)

        # Average over time (excluding S_0 at column 0)
        dA_dsigma = np.mean(dS_dsigma[:, 1:], axis=1)  # (N_paths,)

        # Only count paths where average > strike
        A = np.mean(paths[:, 1:], axis=1)
        in_the_money = (A > K).astype(float)

        return in_the_money * dA_dsigma


# ============================================================================
# LIKELIHOOD RATIO METHOD — Differentiate the density, not the payoff
# ============================================================================
# The likelihood ratio method (also called the Score Function method)
# computes Greeks by differentiating the transition density with respect
# to the parameter, leaving the payoff function untouched.
#
# Formula: Greek = E[ payoff * score ]
# where score = d(log density) / d(parameter)
#
# KEY ADVANTAGE OVER PATHWISE:
# The score function doesn't care if the payoff is discontinuous.
# We can use THE SAME score function for ANY option type — European,
# Asian, barrier, digital — as long as the underlying follows the
# same stochastic process. The payoff just gets multiplied by the score.
#
# WHY IT HAS HIGHER VARIANCE:
# The score function can take extreme values in the tails of the
# distribution. For example, if the parameter is S0 and the path
# ends far from S0, the score can be very large or very small.
# This inflates the variance compared to pathwise.
#
# This tradeoff — generality vs variance — is a central result in
# Monte Carlo sensitivity analysis and one that this project
# empirically demonstrates.
# ============================================================================

class LikelihoodRatioGreeks:
    """
    Likelihood ratio (score function) sensitivity estimates.

    This class computes Greeks by weighting each payoff by a score
    function derived from the transition density.

    Works for: ALL option types (European, Asian, Barrier, digital, etc.)
    Cost: Higher variance than pathwise (when pathwise works).
    """

    @staticmethod
    def score_delta_gbm(
        S_T: np.ndarray,
        S0: float,
        sigma: float,
        T: float,
        r: float,
    ) -> np.ndarray:
        """
        Score function for Delta under Geometric Brownian Motion.

        Derivation:
        Under GBM, log(S_T / S0) ~ N( (r - sigma^2/2)*T , sigma^2 * T )

        The transition density of log(S_T) given S0:
        p(s_T | S0) = (1 / (sigma * sqrt(2*pi*T))) *
                       exp( -(s_T - s0 - (r - sigma^2/2)*T)^2 / (2*sigma^2*T) )
        where s_T = log(S_T), s0 = log(S0)

        The score is:
        d(log p) / d(S0) = d(log p) / d(s0) * d(s0) / d(S0)
        d(log p) / d(s0) = (s_T - s0 - (r - sigma^2/2)*T) / (sigma^2 * T)
        d(s0) / d(S0) = 1 / S0

        Therefore:
        score = (log(S_T/S0) - (r - sigma^2/2)*T) / (S0 * sigma^2 * T)

        This has an intuitive interpretation:
        - If S_T >> S0 (large positive return), score is positive:
          increasing S0 would have made this outcome MORE likely.
        - If S_T << S0 (large negative return), score is negative:
          increasing S0 would have made this outcome LESS likely.

        Parameters
        ----------
        S_T : np.ndarray of shape (N_paths,)
            Terminal asset prices.
        S0 : float
            Initial asset price.
        sigma : float
            Volatility.
        T : float
            Time to maturity.
        r : float
            Risk-free rate.

        Returns
        -------
        score : np.ndarray of shape (N_paths,)
            Score function values for each path.
        """
        log_return = np.log(S_T / S0)
        drift = (r - 0.5 * sigma ** 2) * T

        return (log_return - drift) / (S0 * sigma ** 2 * T)

    @staticmethod
    def score_vega_gbm(
        S_T: np.ndarray,
        S0: float,
        sigma: float,
        T: float,
        r: float,
    ) -> np.ndarray:
        """
        Score function for Vega under Geometric Brownian Motion.

        Derivation:
        d(log p) / d(sigma) = ???

        The log-density of s_T = log(S_T) under GBM:
        log p = -log(sigma) - 0.5*log(2*pi*T)
                - (s_T - s0 - (r - sigma^2/2)*T)^2 / (2*sigma^2*T)

        Differentiating with respect to sigma (using the chain rule
        and simplifying):

        score = -1/sigma + (Z^2 - 1)/sigma - Z*sigma*T/sigma
               = (Z^2 - 1 - Z*sigma*sqrt(T)) / sigma

        where Z = (log(S_T/S0) - (r - sigma^2/2)*T) / (sigma * sqrt(T))

        After algebraic simplification:
        score = [ (log(S_T/S0) - (r - sigma^2/2)*T)^2 / (sigma^2 * T) - 1
                  - sigma*sqrt(T) * (log(S_T/S0) - (r - sigma^2/2)*T) /
                    (sigma * sqrt(T)) ] / sigma

        Parameters
        ----------
        S_T : np.ndarray of shape (N_paths,)
            Terminal asset prices.
        S0 : float
            Initial asset price.
        sigma : float
            Volatility.
        T : float
            Time to maturity.
        r : float
            Risk-free rate.

        Returns
        -------
        score : np.ndarray of shape (N_paths,)
            Score function values for each path.
        """
        log_return = np.log(S_T / S0)
        drift = (r - 0.5 * sigma ** 2) * T

        # Standardized log-return
        Z = (log_return - drift) / (sigma * np.sqrt(T))

        # Score for Vega
        # score = (Z^2 - 1 - Z * sigma * sqrt(T)) / sigma
        score = (Z ** 2 - 1.0 - Z * sigma * np.sqrt(T)) / sigma

        return score

    @staticmethod
    def score_delta_heston(
        S_T: np.ndarray,
        S0: float,
        v_T: np.ndarray,
        v0: float,
        T: float,
        r: float,
    ) -> np.ndarray:
        """
        Score function for Delta under the Heston model.

        WARNING: This is an APPROXIMATION. The exact transition density
        of the Heston model involves a non-central chi-squared distribution
        for the variance and is quite complex. We use the GBM score as
        an approximation, which works reasonably well when variance
        doesn't deviate too far from its initial value.

        For a FULL implementation, one would use the characteristic
        function approach (Heston 1993) or Malliavin calculus. This
        simplified version is sufficient for demonstrating the
        likelihood ratio methodology.

        Parameters
        ----------
        S_T : np.ndarray
            Terminal asset prices.
        S0 : float
            Initial asset price.
        v_T : np.ndarray
            Terminal variance (unused in this approximation).
        v0 : float
            Initial variance.
        T : float
            Time to maturity.
        r : float
            Risk-free rate.

        Returns
        -------
        score : np.ndarray
            Approximate score function values for Delta.
        """
        # Use average realized variance as an approximation
        # A full implementation would use the exact Heston density
        sigma_approx = np.sqrt(max(v0, 1e-8))
        return LikelihoodRatioGreeks.score_delta_gbm(
            S_T, S0, sigma_approx, T, r
        )

    @staticmethod
    def delta(
        payoffs: np.ndarray,
        scores: np.ndarray,
        discount_factor: float,
    ) -> float:
        """
        Compute Delta via the likelihood ratio method.

        Delta = E[ payoff * score ] * discount_factor
             ≈ discount * (1/N) * sum(payoff_i * score_i)

        Parameters
        ----------
        payoffs : np.ndarray of shape (N_paths,)
            Raw (undiscounted) payoffs for each path.
        scores : np.ndarray of shape (N_paths,)
            Score function values for each path.
        discount_factor : float
            exp(-rT) to discount to present value.

        Returns
        -------
        delta : float
            Likelihood ratio estimate of Delta.
        """
        return discount_factor * np.mean(payoffs * scores)

    @staticmethod
    def vega(
        payoffs: np.ndarray,
        scores: np.ndarray,
        discount_factor: float,
    ) -> float:
        """
        Compute Vega via the likelihood ratio method.

        Vega = E[ payoff * score ] * discount_factor
             ≈ discount * (1/N) * sum(payoff_i * score_i)

        Parameters
        ----------
        payoffs : np.ndarray of shape (N_paths,)
            Raw (undiscounted) payoffs for each path.
        scores : np.ndarray of shape (N_paths,)
            Score function values for each path.
        discount_factor : float
            exp(-rT) to discount to present value.

        Returns
        -------
        vega : float
            Likelihood ratio estimate of Vega.
        """
        return discount_factor * np.mean(payoffs * scores)


# ============================================================================
# GREEKS COMPARISON — Compute all methods and compare
# ============================================================================

def compute_all_deltas(
    S_T: np.ndarray,
    payoffs: np.ndarray,
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    discount_factor: float,
    price_func: Callable[[float], float],
    fd_h: float = 1e-4,
) -> dict:
    """
    Compute Delta using all available methods and return a comparison.

    This is the main function for the Greeks comparison analysis.
    It computes:
    1. Pathwise Delta (if payoffs are continuous — caller's responsibility)
    2. Likelihood Ratio Delta
    3. Finite Difference Delta
    4. Black-Scholes Delta (if applicable — European option only)

    Parameters
    ----------
    S_T : np.ndarray
        Terminal asset prices.
    payoffs : np.ndarray
        Raw option payoffs for each path.
    S0, K, T, r, sigma : float
        Option and model parameters.
    discount_factor : float
        exp(-rT).
    price_func : callable
        Function S0 -> price for finite difference.
    fd_h : float
        Perturbation size for finite difference.

    Returns
    -------
    results : dict
        Dictionary with Delta estimates from each method.
    """
    results = {}

    # Pathwise Delta
    pw_delta = PathwiseGreeks.delta_european(S_T, S0, K, T, r)
    results['pathwise'] = discount_factor * np.mean(pw_delta)

    # Likelihood Ratio Delta
    lr_scores = LikelihoodRatioGreeks.score_delta_gbm(S_T, S0, sigma, T, r)
    results['likelihood_ratio'] = LikelihoodRatioGreeks.delta(
        payoffs, lr_scores, discount_factor
    )

    # Finite Difference Delta
    results['finite_difference'] = finite_difference_delta(
        price_func, S0, fd_h
    )

    # Black-Scholes closed-form Delta
    results['black_scholes'] = black_scholes_delta(S0, K, T, r, sigma)

    return results


def compute_all_vegas(
    S_T: np.ndarray,
    payoffs: np.ndarray,
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    discount_factor: float,
    price_func: Callable[[float], float],
    fd_h: float = 1e-4,
) -> dict:
    """
    Compute Vega using all available methods and return a comparison.

    Parameters
    ----------
    S_T : np.ndarray
        Terminal asset prices.
    payoffs : np.ndarray
        Raw option payoffs for each path.
    S0, K, T, r, sigma : float
        Option and model parameters.
    discount_factor : float
        exp(-rT).
    price_func : callable
        Function sigma -> price for finite difference.
    fd_h : float
        Perturbation size for finite difference.

    Returns
    -------
    results : dict
        Dictionary with Vega estimates from each method.
    """
    results = {}

    # Pathwise Vega
    pw_vega = PathwiseGreeks.vega_european(S_T, S0, K, T, r, sigma)
    results['pathwise'] = discount_factor * np.mean(pw_vega)

    # Likelihood Ratio Vega
    lr_scores = LikelihoodRatioGreeks.score_vega_gbm(S_T, S0, sigma, T, r)
    results['likelihood_ratio'] = LikelihoodRatioGreeks.vega(
        payoffs, lr_scores, discount_factor
    )

    # Finite Difference Vega
    results['finite_difference'] = finite_difference_vega(
        price_func, sigma, fd_h
    )

    # Black-Scholes closed-form Vega
    results['black_scholes'] = black_scholes_vega(S0, K, T, r, sigma)

    return results