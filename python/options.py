"""
Option Payoff Classes — European, Asian, and Barrier
======================================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/monte-carlo-option-pricer

This module implements the three option types required by the project.
Each class encapsulates the payoff logic and provides a method to compute
the discounted expected payoff from simulated paths.

OPTION TYPES IMPLEMENTED
-------------------------
1. EuropeanOption
   Pays max(S_T - K, 0) at maturity. Only the terminal value matters.
   The simplest option. Closed-form Black-Scholes price available for
   validation. Used to verify the Monte Carlo engine is correct.

2. AsianOption (Arithmetic Average)
   Pays max(average(S_t) - K, 0) at maturity. The entire path matters
   because the payoff depends on the arithmetic mean of all monitoring
   dates. Asian options are cheaper than European options because the
   averaging reduces volatility. No simple closed-form price exists
   (except under geometric averaging, which we don't use).

3. BarrierOption (Up-and-Out Call)
   Pays max(S_T - K, 0) BUT the option is knocked out (becomes worthless)
   if the underlying ever trades AT OR ABOVE a barrier level B > S0
   during the life of the option. Barrier options are cheaper than
   vanilla options because there's a chance of knockout. They are
   "path-dependent" in the strongest sense — every point along the path
   must be monitored.

DESIGN PRINCIPLES
-----------------
- Each option class is independent and stateless (aside from parameters).
  You create it once and call price() many times with different paths.
- Payoff computation is vectorized over paths using NumPy. No loops.
- All prices are DISCOUNTED to present value using the risk-free rate.
- The barrier condition uses np.any() with axis=1 for efficient
  monitoring across all time steps simultaneously.

References
----------
Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
    Springer. Chapters 1, 3, and 4.
Hull, J. (2022). Options, Futures, and Other Derivatives (11th ed.).
    Pearson. Chapters 15 and 26.
"""

import numpy as np
from typing import Optional, Tuple


# ============================================================================
# EUROPEAN OPTION — The "Hello World" of option pricing
# ============================================================================
# A European call option gives the holder the right (but not the obligation)
# to buy the underlying asset at the strike price K on the expiration date T.
#
# Payoff at maturity:
#     max(S_T - K, 0)
#
# This is the simplest option because only the terminal price matters.
# We can use it to:
# 1. Validate our Monte Carlo engine against the Black-Scholes formula.
# 2. Test Greeks implementations (pathwise and likelihood ratio).
# 3. Establish baseline performance before moving to path-dependent options.
#
# The Black-Scholes closed-form price for a European call:
#     C = S_0 * N(d1) - K * exp(-rT) * N(d2)
#     d1 = [ln(S_0/K) + (r + sigma^2/2)*T] / (sigma * sqrt(T))
#     d2 = d1 - sigma * sqrt(T)
# where N(·) is the standard normal CDF.
# ============================================================================

class EuropeanOption:
    """
    European call option.

    Parameters
    ----------
    K : float
        Strike price. The price at which the holder can buy the asset.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate, continuously compounded.

    Examples
    --------
    >>> option = EuropeanOption(K=100.0, T=1.0, r=0.05)
    >>> paths = np.array([[90, 95, 105], [80, 85, 90]])  # (N_paths, N_steps+1)
    >>> price = option.price_from_terminal(paths[:, -1])  # Uses only S_T
    """

    def __init__(self, K: float, T: float, r: float) -> None:
        """
        Parameters
        ----------
        K : float
            Strike price. Must be > 0.
        T : float
            Time to maturity in years. Must be > 0.
        r : float
            Risk-free rate, continuously compounded.
        """
        if K <= 0:
            raise ValueError(f"Strike K must be positive, got {K}")
        if T <= 0:
            raise ValueError(f"Maturity T must be positive, got {T}")

        self.K = K
        self.T = T
        self.r = r

        # Precompute discount factor: exp(-rT)
        # This is the present value of $1 received at time T.
        # All payoffs are multiplied by this factor to get the
        # risk-neutral present value.
        self._discount = np.exp(-self.r * self.T)

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """
        Compute the raw (undiscounted) payoff for each path.

        Parameters
        ----------
        S_T : np.ndarray of shape (N_paths,)
            Terminal asset prices at maturity T.

        Returns
        -------
        payoff : np.ndarray of shape (N_paths,)
            max(S_T - K, 0) for each path.
        """
        return np.maximum(S_T - self.K, 0.0)

    def price_from_terminal(self, S_T: np.ndarray) -> float:
        """
        Price the option from terminal asset prices.

        This is the standard Monte Carlo estimator:
            price = exp(-rT) * (1/N) * sum(max(S_T^i - K, 0))

        Parameters
        ----------
        S_T : np.ndarray of shape (N_paths,)
            Terminal asset prices from Monte Carlo simulation.

        Returns
        -------
        price : float
            Discounted expected payoff (Monte Carlo estimate).
        """
        payoffs = self.payoff(S_T)
        return self._discount * np.mean(payoffs)

    def price_from_paths(self, paths: np.ndarray) -> float:
        """
        Price the option from full paths (uses only terminal value).

        Convenience method for API consistency with path-dependent options.
        For European options, only paths[:, -1] is used.

        Parameters
        ----------
        paths : np.ndarray of shape (N_paths, N_steps + 1)
            Full asset price paths.

        Returns
        -------
        price : float
            Discounted expected payoff.
        """
        return self.price_from_terminal(paths[:, -1])


# ============================================================================
# ASIAN OPTION — Arithmetic Average Rate Call
# ============================================================================
# An Asian call option pays off based on the AVERAGE price of the underlying
# asset over the life of the option, not just the terminal price.
#
# Payoff at maturity:
#     max(A - K, 0)
# where A = (1/n) * sum_{i=1}^{n} S_{t_i}
#
# KEY INSIGHT: Averaging reduces volatility. The average of a random process
# has lower variance than the process itself. This makes Asian options
# CHEAPER than equivalent European options.
#
# WHY ASIAN OPTIONS EXIST:
# - Commodity markets: An oil importer cares about the average price paid
#   over a quarter, not the price on a single day.
# - FX markets: A multinational corporation hedging currency exposure over
#   a fiscal year needs average-rate protection.
# - They're harder to manipulate: It's expensive to manipulate the average
#   vs a single closing price.
#
# WHY THEY'RE HARDER TO PRICE:
# - Path-dependent: Need full trajectories, not just S_T.
# - No closed-form solution: The arithmetic average of lognormal random
#   variables is not lognormal. Approximation formulas exist (Turnbull-Wakeman)
#   but Monte Carlo is the standard approach.
# - Greeks are non-trivial: Pathwise method works (payoff is continuous
#   in the parameters) but requires differentiating through the average.
#
# NOTE: We use arithmetic averaging. The geometric average DOES have a
# closed-form solution (because product of lognormals is lognormal) but
# real Asian options use arithmetic averaging.
# ============================================================================

class AsianOption:
    """
    Arithmetic average rate Asian call option.

    Parameters
    ----------
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate, continuously compounded.

    Notes
    -----
    The average is computed over ALL time steps in the path.
    The monitoring frequency is determined by N_steps in the simulation:
        n = N_steps
        t_i = i * T / N_steps  for i = 1, ..., n
        A = (1/n) * sum(S_{t_i})

    Some Asian options exclude an initial "averaging period" or average
    over a subset of dates. This implementation averages over all
    simulated dates for simplicity.
    """

    def __init__(self, K: float, T: float, r: float) -> None:
        """
        Parameters
        ----------
        K : float
            Strike price. Must be > 0.
        T : float
            Time to maturity in years. Must be > 0.
        r : float
            Risk-free rate.
        """
        if K <= 0:
            raise ValueError(f"Strike K must be positive, got {K}")
        if T <= 0:
            raise ValueError(f"Maturity T must be positive, got {T}")

        self.K = K
        self.T = T
        self.r = r
        self._discount = np.exp(-self.r * self.T)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute the raw payoff for each path.

        The arithmetic average is computed along axis=1 (across time steps
        for each path). We average all columns EXCEPT column 0 (S_0) because
        the initial price is known and not part of the option's averaging.

        Parameters
        ----------
        paths : np.ndarray of shape (N_paths, N_steps + 1)
            Full asset price paths. Column 0 = S_0.

        Returns
        -------
        payoff : np.ndarray of shape (N_paths,)
            max(average(S_t) - K, 0) for each path.
        """
        # Compute arithmetic mean along the time axis
        # We exclude S_0 (column 0) from the average because the option
        # averages over future prices. Including S_0 would slightly
        # reduce variance and make the option cheaper — some contracts
        # do include it, but the standard convention excludes it.
        # If paths include S_0 at column 0 and N_steps monitoring dates
        # after that, the average is over columns 1 to N_steps (inclusive).
        average_price = np.mean(paths[:, 1:], axis=1)

        return np.maximum(average_price - self.K, 0.0)

    def price(self, paths: np.ndarray) -> float:
        """
        Price the Asian option from simulated paths.

        Parameters
        ----------
        paths : np.ndarray of shape (N_paths, N_steps + 1)
            Full asset price paths from Monte Carlo simulation.

        Returns
        -------
        price : float
            Discounted expected payoff (Monte Carlo estimate).
        """
        payoffs = self.payoff(paths)
        return self._discount * np.mean(payoffs)


# ============================================================================
# BARRIER OPTION — Up-and-Out Call
# ============================================================================
# A barrier option is activated or extinguished if the underlying asset
# reaches a predetermined barrier level during the option's life.
#
# UP-AND-OUT CALL:
# - Starts as a regular European call.
# - If the underlying ever trades AT OR ABOVE the barrier B, the option
#   is "knocked out" and becomes WORTHLESS, even if it later drops below B.
# - If the barrier is never touched, payoff is max(S_T - K, 0).
#
# WHY BARRIER OPTIONS EXIST:
# - Cheaper than vanilla options (knockout risk → lower premium).
# - Useful for investors with a specific view: "I think the stock will go
#   up, but not above B."
# - Common in FX markets where central bank intervention creates natural
#   barriers.
# - Embedded in structured products (reverse convertibles, autocallables).
#
# WHY THEY'RE THE HARDEST TO PRICE:
# - Strongly path-dependent: Every point on the path matters.
# - Discrete monitoring bias: In reality, barriers are monitored
#   continuously, but Monte Carlo monitors at discrete time steps.
#   This creates a bias: discrete monitoring misses some barrier hits
#   that occur between observation dates, OVERPRICING the option.
#   We document this in the convergence analysis.
# - Greeks are discontinuous: A tiny parameter change can flip a path
#   from "hits barrier" to "doesn't hit barrier", causing a discrete
#   jump in payoff. This breaks the pathwise method — one of the key
#   demonstrations in this project.
#
# MONITORING CONVENTION:
# We monitor at EVERY simulated time step. The barrier is hit if
# max(S_t) >= B for any t in {t_0, t_1, ..., t_n}.
# Some conventions monitor only at closing prices or exclude the initial
# date. We monitor all dates including t_0 for simplicity, but since
# S_0 < B by assumption (B > S_0 for up-and-out), this doesn't affect
# the result.
# ============================================================================

class BarrierOption:
    """
    Up-and-out barrier call option.

    The option is knocked out (becomes worthless) if the underlying
    asset price ever reaches or exceeds the barrier level B during
    the life of the option.

    Parameters
    ----------
    K : float
        Strike price.
    B : float
        Barrier level. Must be > S0 (for up-and-out).
        If B <= S0, the option is immediately knocked out at t=0.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate, continuously compounded.

    Notes
    -----
    The barrier condition is checked at every simulated time step:
        knocked_out = any S_t >= B for t in [t_0, t_1, ..., t_N]

    This is discrete monitoring. With N_steps -> infinity, the
    discrete monitoring price converges to the continuous monitoring
    price FROM ABOVE (because we miss more crossings with fewer steps).
    """

    def __init__(self, K: float, B: float, T: float, r: float) -> None:
        """
        Parameters
        ----------
        K : float
            Strike price. Must be > 0.
        B : float
            Barrier level. Must be > 0.
        T : float
            Time to maturity in years. Must be > 0.
        r : float
            Risk-free rate.
        """
        if K <= 0:
            raise ValueError(f"Strike K must be positive, got {K}")
        if B <= 0:
            raise ValueError(f"Barrier B must be positive, got {B}")
        if T <= 0:
            raise ValueError(f"Maturity T must be positive, got {T}")

        self.K = K
        self.B = B
        self.T = T
        self.r = r
        self._discount = np.exp(-self.r * self.T)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute the raw payoff for each path, accounting for knockout.

        For each path:
        1. Check if max(S_t) >= B at any monitoring date.
           If yes -> payoff = 0 (knocked out).
        2. Otherwise -> payoff = max(S_T - K, 0) (surviving call).

        Parameters
        ----------
        paths : np.ndarray of shape (N_paths, N_steps + 1)
            Full asset price paths.

        Returns
        -------
        payoff : np.ndarray of shape (N_paths,)
            Payoff for each path after barrier condition.
        """
        # ------------------------------------------------------------------
        # Detect barrier hits: any(S_t >= B) along axis=1 (time axis)
        #
        # np.any(..., axis=1) returns True for paths where at least one
        # monitoring date has S_t >= B.
        #
        # Shape: (N_paths,)
        # ------------------------------------------------------------------
        knocked_out = np.any(paths >= self.B, axis=1)

        # ------------------------------------------------------------------
        # Compute European payoff for all paths
        # Shape: (N_paths,)
        # ------------------------------------------------------------------
        european_payoff = np.maximum(paths[:, -1] - self.K, 0.0)

        # ------------------------------------------------------------------
        # Zero out knocked-out paths
        # np.where(condition, x, y) returns x where condition is True,
        # y where condition is False.
        # ------------------------------------------------------------------
        return np.where(knocked_out, 0.0, european_payoff)

    def price(self, paths: np.ndarray) -> float:
        """
        Price the barrier option from simulated paths.

        Parameters
        ----------
        paths : np.ndarray of shape (N_paths, N_steps + 1)
            Full asset price paths from Monte Carlo simulation.

        Returns
        -------
        price : float
            Discounted expected payoff (Monte Carlo estimate).
        """
        payoffs = self.payoff(paths)
        return self._discount * np.mean(payoffs)

    @property
    def knockout_probability_estimate(self) -> Optional[float]:
        """
        Estimate the probability of knockout from the most recent pricing.
        Not available until price() has been called at least once.

        This is stored as an attribute by the price() method.
        """
        return getattr(self, '_knockout_prob', None)

    def price_with_knockout_info(self, paths: np.ndarray) -> Tuple[float, float]:
        """
        Price the option and also return the estimated knockout probability.

        The knockout probability is useful for:
        - Understanding the risk profile of the option.
        - Validating that the barrier is being tested as expected.
        - Debugging the Greeks (pathwise fails because of this discontinuity).

        Parameters
        ----------
        paths : np.ndarray of shape (N_paths, N_steps + 1)
            Full asset price paths.

        Returns
        -------
        price : float
            Discounted expected payoff.
        knockout_prob : float
            Fraction of paths that hit the barrier.
        """
        knocked_out = np.any(paths >= self.B, axis=1)
        european_payoff = np.maximum(paths[:, -1] - self.K, 0.0)
        payoffs = np.where(knocked_out, 0.0, european_payoff)

        knockout_prob = np.mean(knocked_out)
        self._knockout_prob = knockout_prob

        price = self._discount * np.mean(payoffs)
        return price, knockout_prob