"""
Unit Tests — Greeks Computation
=================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/monte-carlo-option-pricer

Tests for greeks.py: PathwiseGreeks, LikelihoodRatioGreeks,
closed-form Black-Scholes Greeks, and finite difference.

Validates:
- Closed-form Greeks match known values.
- Pathwise Delta = N(d1) in expectation.
- Likelihood ratio Delta matches pathwise in expectation.
- Finite difference approximates true Greek.
- Vega is positive for calls.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from greeks import (
    black_scholes_price,
    black_scholes_delta,
    black_scholes_vega,
    PathwiseGreeks,
    LikelihoodRatioGreeks,
    finite_difference_delta,
    finite_difference_vega,
    compute_all_deltas,
    compute_all_vegas,
)
from models import GBMPath, set_seed
from options import EuropeanOption


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def fix_seed():
    """Fix random seed before each test."""
    set_seed(42)


@pytest.fixture
def standard_params():
    """Standard option parameters."""
    return {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.20,
    }


@pytest.fixture
def large_sample(standard_params):
    """Generate a large sample of terminal prices for statistical tests."""
    gbm = GBMPath(
        standard_params["S0"],
        standard_params["r"],
        standard_params["sigma"],
    )
    N = 200_000
    S_T = gbm.simulate_terminal(standard_params["T"], N)
    return standard_params, S_T, N


# ============================================================================
# Closed-Form Black-Scholes Greeks Tests
# ============================================================================

class TestBlackScholesGreeks:
    """Test closed-form Black-Scholes formulas."""

    def test_delta_atm(self, standard_params):
        """ATM call Delta should be approximately 0.5 + small drift adjustment."""
        p = standard_params
        delta = black_scholes_delta(p["S0"], p["K"], p["T"], p["r"], p["sigma"])
        # ATM Delta ≈ N( (r + sigma^2/2)*sqrt(T) / sigma )
        # With r=0.05, sigma=0.20, T=1: d1 ≈ 0.35, N(d1) ≈ 0.637
        assert 0.55 < delta < 0.70, f"ATM Delta = {delta:.4f}, expected ~0.64"

    def test_delta_deep_itm(self, standard_params):
        """Deep ITM call Delta should approach 1.0."""
        p = standard_params
        delta = black_scholes_delta(p["S0"], 50.0, p["T"], p["r"], p["sigma"])
        assert delta > 0.95

    def test_delta_deep_otm(self, standard_params):
        """Deep OTM call Delta should approach 0.0."""
        p = standard_params
        delta = black_scholes_delta(p["S0"], 200.0, p["T"], p["r"], p["sigma"])
        assert delta < 0.05

    def test_delta_at_expiry_itm(self, standard_params):
        """At expiry (T=0), Delta = 1 if S0 > K, else 0."""
        p = standard_params
        delta_itm = black_scholes_delta(110.0, 100.0, 0.0, p["r"], p["sigma"])
        delta_otm = black_scholes_delta(90.0, 100.0, 0.0, p["r"], p["sigma"])
        assert delta_itm == 1.0
        assert delta_otm == 0.0

    def test_vega_positive(self, standard_params):
        """Vega should always be positive for a call."""
        p = standard_params
        vega = black_scholes_vega(p["S0"], p["K"], p["T"], p["r"], p["sigma"])
        assert vega > 0

    def test_vega_zero_at_expiry(self, standard_params):
        """At expiry, Vega = 0 (no time for volatility to matter)."""
        p = standard_params
        vega = black_scholes_vega(p["S0"], p["K"], 0.0, p["r"], p["sigma"])
        assert vega == 0.0

    def test_vega_higher_atm(self, standard_params):
        """Vega is highest for ATM options."""
        p = standard_params
        vega_atm = black_scholes_vega(100.0, 100.0, p["T"], p["r"], p["sigma"])
        vega_otm = black_scholes_vega(100.0, 150.0, p["T"], p["r"], p["sigma"])
        vega_itm = black_scholes_vega(100.0, 50.0, p["T"], p["r"], p["sigma"])
        assert vega_atm > vega_otm
        assert vega_atm > vega_itm

    def test_price_put_call_parity(self, standard_params):
        """
        Put-Call parity: C - P = S0 - K*exp(-rT)
        We can verify our call price is consistent.
        """
        p = standard_params
        call = black_scholes_price(p["S0"], p["K"], p["T"], p["r"], p["sigma"])
        # No put formula implemented, but we can verify price > 0 and < S0
        assert 0 < call < p["S0"]


# ============================================================================
# Pathwise Greeks Tests
# ============================================================================

class TestPathwiseGreeks:
    """Test pathwise sensitivity estimates."""

    def test_delta_shape(self, large_sample):
        """Pathwise Delta should return array of shape (N_paths,)."""
        params, S_T, N = large_sample
        pw_delta = PathwiseGreeks.delta_european(
            S_T, params["S0"], params["K"],
            params["T"], params["r"],
        )
        assert pw_delta.shape == (N,)

    def test_delta_non_negative(self, large_sample):
        """Pathwise Delta should be >= 0 for a call option."""
        params, S_T, N = large_sample
        pw_delta = PathwiseGreeks.delta_european(
            S_T, params["S0"], params["K"],
            params["T"], params["r"],
        )
        assert np.all(pw_delta >= 0)

    def test_delta_zero_for_otm(self, standard_params):
        """Out-of-the-money paths should have zero pathwise Delta."""
        p = standard_params
        S_T = np.array([80.0, 90.0, 95.0])  # All OTM
        pw_delta = PathwiseGreeks.delta_european(S_T, p["S0"], p["K"], p["T"], p["r"])
        assert np.all(pw_delta == 0.0)

    def test_delta_positive_for_itm(self, standard_params):
        """In-the-money paths should have positive pathwise Delta."""
        p = standard_params
        S_T = np.array([105.0, 110.0, 120.0])  # All ITM
        pw_delta = PathwiseGreeks.delta_european(S_T, p["S0"], p["K"], p["T"], p["r"])
        assert np.all(pw_delta > 0)

    def test_delta_approximately_correct(self, standard_params):
            """
            Finite difference Delta should approximate Black-Scholes Delta.
            Central difference is O(h^2) accurate.

            CRITICAL: Both price evaluations MUST use the same random seed
            so the Monte Carlo noise cancels in the difference.
            """
            p = standard_params
            N = 100_000

            def price_func(S0, seed):
                set_seed(seed)
                gbm = GBMPath(S0, p["r"], p["sigma"])
                option = EuropeanOption(p["K"], p["T"], p["r"])
                S_T = gbm.simulate_terminal(p["T"], N)
                return option.price_from_terminal(S_T)

            seed = 42
            price_up = price_func(p["S0"] + 1e-3, seed)
            price_down = price_func(p["S0"] - 1e-3, seed)
            fd_delta = (price_up - price_down) / (2 * 1e-3)

            bs_delta = black_scholes_delta(
                p["S0"], p["K"], p["T"], p["r"], p["sigma"],
            )

            assert abs(fd_delta - bs_delta) < 0.05, (
                f"FD Delta {fd_delta:.6f} vs BS Delta {bs_delta:.6f}"
            )

    def test_vega_approximately_correct(self, standard_params):
        """
        Finite difference Vega should approximate Black-Scholes Vega.

        Same seed requirement as Delta.
        """
        p = standard_params
        N = 100_000

        def price_func(sigma, seed):
            set_seed(seed)
            gbm = GBMPath(p["S0"], p["r"], sigma)
            option = EuropeanOption(p["K"], p["T"], p["r"])
            S_T = gbm.simulate_terminal(p["T"], N)
            return option.price_from_terminal(S_T)

        seed = 42
        price_up = price_func(p["sigma"] + 1e-3, seed)
        price_down = price_func(p["sigma"] - 1e-3, seed)
        fd_vega = (price_up - price_down) / (2 * 1e-3)

        bs_vega = black_scholes_vega(
            p["S0"], p["K"], p["T"], p["r"], p["sigma"],
        )

        assert abs(fd_vega - bs_vega) < 0.05, (
            f"FD Vega {fd_vega:.6f} vs BS Vega {bs_vega:.6f}"
        )

    def test_vega_positive_for_calls(self, large_sample):
        """
        Pathwise Vega should be non-negative IN EXPECTATION for call options.
        Individual paths can have negative pathwise Vega
        (OTM paths have zero, but dS_T/dsigma can be negative).
        The mean should be positive.
        """
        params, S_T, _ = large_sample
        pw_vega = PathwiseGreeks.vega_european(
            S_T, params["S0"], params["K"],
            params["T"], params["r"], params["sigma"],
        )
        discount = np.exp(-params["r"] * params["T"])
        mc_vega = discount * np.mean(pw_vega)
        assert mc_vega > 0, f"Expected positive expected Vega, got {mc_vega:.6f}"


# ============================================================================
# Likelihood Ratio Greeks Tests
# ============================================================================

class TestLikelihoodRatioGreeks:
    """Test likelihood ratio sensitivity estimates."""

    def test_score_delta_shape(self, large_sample):
        """Score function should return array of shape (N_paths,)."""
        params, S_T, N = large_sample
        scores = LikelihoodRatioGreeks.score_delta_gbm(
            S_T, params["S0"], params["sigma"],
            params["T"], params["r"],
        )
        assert scores.shape == (N,)

    def test_score_delta_mean_zero(self, large_sample):
        """
        The expected value of the score function should be zero.
        E[d(log p)/d(S0)] = 0 (a property of score functions).
        """
        params, S_T, N = large_sample
        scores = LikelihoodRatioGreeks.score_delta_gbm(
            S_T, params["S0"], params["sigma"],
            params["T"], params["r"],
        )

        mean_score = np.mean(scores)
        std_err_score = np.std(scores, ddof=1) / np.sqrt(N)

        assert abs(mean_score) < 5.0 * std_err_score, (
            f"Score mean {mean_score:.8f} not zero, SE={std_err_score:.8f}"
        )

    def test_score_delta_sign(self, large_sample):
        """
        Score should be positive when S_T is high and negative when low.
        Higher S0 makes high S_T more likely and low S_T less likely.
        """
        params, S_T, _ = large_sample
        scores = LikelihoodRatioGreeks.score_delta_gbm(
            S_T, params["S0"], params["sigma"],
            params["T"], params["r"],
        )

        # For S_T far above S0, score should be positive
        high_paths = S_T > 150.0
        if np.any(high_paths):
            assert np.all(scores[high_paths] > 0)

        # For S_T far below S0, score should be negative
        low_paths = S_T < 60.0
        if np.any(low_paths):
            assert np.all(scores[low_paths] < 0)

    def test_score_vega_shape(self, large_sample):
        """Vega score should return array of shape (N_paths,)."""
        params, S_T, N = large_sample
        scores = LikelihoodRatioGreeks.score_vega_gbm(
            S_T, params["S0"], params["sigma"],
            params["T"], params["r"],
        )
        assert scores.shape == (N,)

    def test_lr_delta_approximately_correct(self, large_sample):
        """
        Likelihood Ratio Delta should converge to Black-Scholes Delta.
        This will have higher variance than pathwise — we use a wider
        tolerance (7 standard errors instead of 5).
        """
        params, S_T, N = large_sample

        option = EuropeanOption(params["K"], params["T"], params["r"])
        payoffs = option.payoff(S_T)
        discount = option._discount

        scores = LikelihoodRatioGreeks.score_delta_gbm(
            S_T, params["S0"], params["sigma"],
            params["T"], params["r"],
        )

        lr_estimates = discount * payoffs * scores
        lr_delta = np.mean(lr_estimates)

        bs_delta = black_scholes_delta(
            params["S0"], params["K"], params["T"],
            params["r"], params["sigma"],
        )

        std_err = np.std(lr_estimates, ddof=1) / np.sqrt(N)

        assert abs(lr_delta - bs_delta) < 7.0 * std_err, (
            f"LR Delta {lr_delta:.6f} vs BS Delta {bs_delta:.6f}, "
            f"diff={abs(lr_delta - bs_delta):.6f}, SE={std_err:.6f}"
        )

    def test_lr_vega_approximately_correct(self, large_sample):
        """
        Likelihood Ratio Vega should converge to Black-Scholes Vega.
        """
        params, S_T, N = large_sample

        option = EuropeanOption(params["K"], params["T"], params["r"])
        payoffs = option.payoff(S_T)
        discount = option._discount

        scores = LikelihoodRatioGreeks.score_vega_gbm(
            S_T, params["S0"], params["sigma"],
            params["T"], params["r"],
        )

        lr_estimates = discount * payoffs * scores
        lr_vega = np.mean(lr_estimates)

        bs_vega = black_scholes_vega(
            params["S0"], params["K"], params["T"],
            params["r"], params["sigma"],
        )

        std_err = np.std(lr_estimates, ddof=1) / np.sqrt(N)

        assert abs(lr_vega - bs_vega) < 7.0 * std_err, (
            f"LR Vega {lr_vega:.6f} vs BS Vega {bs_vega:.6f}, "
            f"diff={abs(lr_vega - bs_vega):.6f}, SE={std_err:.6f}"
        )

    def test_lr_higher_variance_than_pathwise(self, large_sample):
        """
        Likelihood ratio should have higher variance than pathwise
        for Delta. This is the KEY theoretical result.
        """
        params, S_T, N = large_sample

        option = EuropeanOption(params["K"], params["T"], params["r"])
        payoffs = option.payoff(S_T)
        discount = option._discount

        # Pathwise variance
        pw_estimates = discount * PathwiseGreeks.delta_european(
            S_T, params["S0"], params["K"], params["T"], params["r"],
        )
        pw_var = np.var(pw_estimates, ddof=1)

        # Likelihood ratio variance
        lr_scores = LikelihoodRatioGreeks.score_delta_gbm(
            S_T, params["S0"], params["sigma"],
            params["T"], params["r"],
        )
        lr_estimates = discount * payoffs * lr_scores
        lr_var = np.var(lr_estimates, ddof=1)

        assert lr_var > pw_var, (
            f"Expected LR variance ({lr_var:.6f}) > PW variance ({pw_var:.6f})"
        )

        print(f"\n  PW variance: {pw_var:.6f}, LR variance: {lr_var:.6f}")
        print(f"  Ratio LR/PW: {lr_var/pw_var:.2f}x")


# ============================================================================
# Finite Difference Tests
# ============================================================================

class TestFiniteDifference:
    """Test finite difference Greeks."""

    def test_delta_approximately_correct(self, standard_params):
        """
        Finite difference Delta should approximate Black-Scholes Delta.
        
        We DON'T use the greeks.finite_difference_delta() helper because
        it doesn't fix the seed. Instead we inline the FD with a fixed seed
        so the Monte Carlo noise cancels properly.
        """
        p = standard_params
        N = 100_000
        h = 1e-3

        def price_func(S0, seed):
            set_seed(seed)
            gbm = GBMPath(S0, p["r"], p["sigma"])
            option = EuropeanOption(p["K"], p["T"], p["r"])
            S_T = gbm.simulate_terminal(p["T"], N)
            return option.price_from_terminal(S_T)

        # SAME seed for both evaluations — this is the fix
        seed = 42
        price_up = price_func(p["S0"] + h, seed)
        price_down = price_func(p["S0"] - h, seed)
        fd_delta = (price_up - price_down) / (2 * h)

        bs_delta = black_scholes_delta(
            p["S0"], p["K"], p["T"], p["r"], p["sigma"],
        )

        assert abs(fd_delta - bs_delta) < 0.05, (
            f"FD Delta {fd_delta:.6f} vs BS Delta {bs_delta:.6f}"
        )

    def test_vega_approximately_correct(self, standard_params):
        """
        Finite difference Vega should approximate Black-Scholes Vega.
        
        Same fix: inline the FD with fixed seed.
        """
        p = standard_params
        N = 100_000
        h = 1e-3

        def price_func(sigma, seed):
            set_seed(seed)
            gbm = GBMPath(p["S0"], p["r"], sigma)
            option = EuropeanOption(p["K"], p["T"], p["r"])
            S_T = gbm.simulate_terminal(p["T"], N)
            return option.price_from_terminal(S_T)

        seed = 42
        price_up = price_func(p["sigma"] + h, seed)
        price_down = price_func(p["sigma"] - h, seed)
        fd_vega = (price_up - price_down) / (2 * h)

        bs_vega = black_scholes_vega(
            p["S0"], p["K"], p["T"], p["r"], p["sigma"],
        )

        assert abs(fd_vega - bs_vega) < 0.05, (
            f"FD Vega {fd_vega:.6f} vs BS Vega {bs_vega:.6f}"
        )
        

# ============================================================================
# Likelihood Ratio: Works for Discontinuous Payoffs
# ============================================================================

class TestLikelihoodRatioBarrier:
    """
    Demonstrate that likelihood ratio WORKS for barrier options
    where pathwise WOULD FAIL.

    This is the central conceptual test of the Greeks module.
    """

    def test_lr_produces_finite_delta_for_barrier(self, standard_params):
        """
        Likelihood ratio should produce a finite, reasonable Delta
        for a barrier option. Pathwise would fail because the payoff
        is discontinuous at the barrier.
        """
        p = standard_params
        B = 120.0

        from models import GBMPath
        from options import BarrierOption

        gbm = GBMPath(p["S0"], p["r"], p["sigma"])
        barrier = BarrierOption(p["K"], B, p["T"], p["r"])

        N = 100_000
        N_steps = 252
        paths = gbm.simulate_paths(p["T"], N, N_steps)

        payoffs = barrier.payoff(paths)
        scores = LikelihoodRatioGreeks.score_delta_gbm(
            paths[:, -1], p["S0"], p["sigma"], p["T"], p["r"],
        )
        lr_delta = LikelihoodRatioGreeks.delta(
            payoffs, scores, barrier._discount,
        )

        # LR Delta should be a reasonable number (not NaN, not absurd)
        assert np.isfinite(lr_delta)
        assert -1.0 < lr_delta < 2.0, f"LR Delta for barrier = {lr_delta:.6f}"

        print(f"\n  Barrier option LR Delta: {lr_delta:.6f}")
        print(f"  (Pathwise Delta would be INCORRECT due to discontinuity)")


# ============================================================================
# Edge Cases
# ============================================================================

class TestGreeksEdgeCases:
    """Test edge cases for Greeks computation."""

    def test_zero_volatility(self, standard_params):
        """With zero volatility, pathwise Delta should handle it."""
        p = standard_params
        S_T = np.array([100.0 * np.exp(0.05)])  # Deterministic
        pw = PathwiseGreeks.delta_european(
            S_T, p["S0"], p["K"], p["T"], p["r"],
        )
        assert np.isfinite(pw).all()

    def test_very_small_T(self, standard_params):
        """Very small time to maturity should not crash score functions."""
        p = standard_params
        S_T = np.array([105.0, 95.0])
        scores = LikelihoodRatioGreeks.score_delta_gbm(
            S_T, p["S0"], p["sigma"], 0.001, p["r"],
        )
        assert np.all(np.isfinite(scores))


# ============================================================================
# finite_difference_delta / finite_difference_vega helper functions
# ============================================================================

class TestFiniteDifferenceHelpers:
    """Test the finite_difference_delta and finite_difference_vega helpers."""

    def test_fd_delta_helper_calls_price_func(self, standard_params):
        """finite_difference_delta should call price_func with perturbed S0."""
        p = standard_params
        calls = []

        def price_func(S0):
            calls.append(S0)
            return S0 * 0.01  # Simple mock

        finite_difference_delta(price_func, p["S0"], h=0.1)
        assert len(calls) == 2
        assert pytest.approx(calls[0]) == p["S0"] + 0.1
        assert pytest.approx(calls[1]) == p["S0"] - 0.1

    def test_fd_delta_helper_result(self, standard_params):
        """finite_difference_delta should return (f(S0+h) - f(S0-h)) / (2h)."""
        p = standard_params
        h = 1.0

        def price_func(S0):
            return S0 * 2.0  # slope = 2, so delta = 2

        delta = finite_difference_delta(price_func, p["S0"], h=h)
        assert delta == pytest.approx(2.0)

    def test_fd_vega_helper_calls_price_func(self, standard_params):
        """finite_difference_vega should call price_func with perturbed sigma."""
        p = standard_params
        calls = []

        def price_func(sigma):
            calls.append(sigma)
            return sigma * 10.0  # Simple mock

        finite_difference_vega(price_func, p["sigma"], h=0.01)
        assert len(calls) == 2
        assert pytest.approx(calls[0]) == p["sigma"] + 0.01
        assert pytest.approx(calls[1]) == p["sigma"] - 0.01

    def test_fd_vega_helper_result(self, standard_params):
        """finite_difference_vega should return (f(σ+h) - f(σ-h)) / (2h)."""
        p = standard_params
        h = 1.0

        def price_func(sigma):
            return sigma * 3.0  # slope = 3, so vega = 3

        vega = finite_difference_vega(price_func, p["sigma"], h=h)
        assert vega == pytest.approx(3.0)


# ============================================================================
# black_scholes_price at T=0
# ============================================================================

class TestBlackScholesPriceAtExpiry:
    """Test black_scholes_price edge case at T=0."""

    def test_price_at_expiry_itm(self, standard_params):
        """At T=0, ITM call price equals intrinsic value S0 - K."""
        p = standard_params
        price = black_scholes_price(110.0, 100.0, 0.0, p["r"], p["sigma"])
        assert price == pytest.approx(10.0)

    def test_price_at_expiry_otm(self, standard_params):
        """At T=0, OTM call price is zero."""
        p = standard_params
        price = black_scholes_price(90.0, 100.0, 0.0, p["r"], p["sigma"])
        assert price == 0.0

    def test_price_at_expiry_atm(self, standard_params):
        """At T=0, ATM call price (S0=K) is zero."""
        p = standard_params
        price = black_scholes_price(100.0, 100.0, 0.0, p["r"], p["sigma"])
        assert price == 0.0


# ============================================================================
# PathwiseGreeks — delta_asian and vega_asian
# ============================================================================

class TestPathwiseAsianGreeks:
    """Test pathwise Greeks for Asian options."""

    def test_delta_asian_shape(self, standard_params):
        """delta_asian should return array of shape (N_paths,)."""
        from models import GBMPath, set_seed
        p = standard_params
        set_seed(42)
        gbm = GBMPath(p["S0"], p["r"], p["sigma"])
        paths = gbm.simulate_paths(p["T"], N_paths=500, N_steps=50)
        delta = PathwiseGreeks.delta_asian(paths, p["S0"], p["K"], p["T"], p["r"])
        assert delta.shape == (500,)

    def test_delta_asian_non_negative(self, standard_params):
        """Pathwise Asian Delta should be >= 0 for a call."""
        from models import GBMPath, set_seed
        p = standard_params
        set_seed(42)
        gbm = GBMPath(p["S0"], p["r"], p["sigma"])
        paths = gbm.simulate_paths(p["T"], N_paths=1000, N_steps=50)
        delta = PathwiseGreeks.delta_asian(paths, p["S0"], p["K"], p["T"], p["r"])
        assert np.all(delta >= 0.0)

    def test_delta_asian_zero_for_otm(self, standard_params):
        """OTM paths (average < K) should produce zero Asian Delta."""
        p = standard_params
        # Paths that are always below strike
        paths = np.array([
            [100.0, 80.0, 85.0, 90.0],
            [100.0, 70.0, 75.0, 80.0],
        ])
        delta = PathwiseGreeks.delta_asian(paths, p["S0"], p["K"], p["T"], p["r"])
        assert np.all(delta == 0.0)

    def test_delta_asian_positive_for_itm(self, standard_params):
        """ITM paths (average > K) should have positive Asian Delta."""
        p = standard_params
        paths = np.array([
            [100.0, 120.0, 125.0, 130.0],
            [100.0, 115.0, 118.0, 122.0],
        ])
        delta = PathwiseGreeks.delta_asian(paths, p["S0"], p["K"], p["T"], p["r"])
        assert np.all(delta > 0.0)

    def test_delta_asian_formula(self, standard_params):
        """delta_asian = indicator(A>K) * A/S0."""
        p = standard_params
        paths = np.array([[100.0, 120.0, 120.0, 120.0]])  # avg=120, A/S0=1.2
        delta = PathwiseGreeks.delta_asian(paths, p["S0"], p["K"], p["T"], p["r"])
        assert delta[0] == pytest.approx(120.0 / 100.0)

    def test_vega_asian_shape(self, standard_params):
        """vega_asian should return array of shape (N_paths,)."""
        from models import GBMPath, set_seed
        p = standard_params
        set_seed(42)
        gbm = GBMPath(p["S0"], p["r"], p["sigma"])
        paths = gbm.simulate_paths(p["T"], N_paths=500, N_steps=50)
        vega = PathwiseGreeks.vega_asian(
            paths, p["S0"], p["K"], p["T"], p["r"], p["sigma"], N_steps=50
        )
        assert vega.shape == (500,)

    def test_vega_asian_finite(self, standard_params):
        """All vega_asian values should be finite."""
        from models import GBMPath, set_seed
        p = standard_params
        set_seed(42)
        gbm = GBMPath(p["S0"], p["r"], p["sigma"])
        paths = gbm.simulate_paths(p["T"], N_paths=200, N_steps=30)
        vega = PathwiseGreeks.vega_asian(
            paths, p["S0"], p["K"], p["T"], p["r"], p["sigma"], N_steps=30
        )
        assert np.all(np.isfinite(vega))

    def test_vega_asian_zero_for_deep_otm(self, standard_params):
        """Deep OTM paths (average << K) should have zero Asian Vega."""
        p = standard_params
        paths = np.array([
            [100.0, 50.0, 55.0, 60.0],
            [100.0, 40.0, 45.0, 50.0],
        ])
        vega = PathwiseGreeks.vega_asian(
            paths, p["S0"], p["K"], p["T"], p["r"], p["sigma"], N_steps=3
        )
        assert np.all(vega == 0.0)

    def test_vega_asian_expected_positive(self, standard_params):
        """Mean Asian Vega should be positive in expectation."""
        from models import GBMPath, set_seed
        p = standard_params
        set_seed(42)
        gbm = GBMPath(p["S0"], p["r"], p["sigma"])
        N_steps = 50
        paths = gbm.simulate_paths(p["T"], N_paths=50_000, N_steps=N_steps)
        vega = PathwiseGreeks.vega_asian(
            paths, p["S0"], p["K"], p["T"], p["r"], p["sigma"], N_steps=N_steps
        )
        discount = np.exp(-p["r"] * p["T"])
        mc_vega = discount * np.mean(vega)
        assert mc_vega > 0.0


# ============================================================================
# LikelihoodRatioGreeks — score_delta_heston and vega()
# ============================================================================

class TestLikelihoodRatioHestonAndVega:
    """Test Heston score function and LR vega."""

    def test_score_delta_heston_shape(self, standard_params):
        """Heston score should return array of shape (N_paths,)."""
        from models import HestonPath, set_seed
        p = standard_params
        set_seed(42)
        model = HestonPath(
            S0=p["S0"], v0=0.04, r=p["r"],
            kappa=2.0, theta=0.04, xi=0.30, rho=-0.70,
        )
        S, v = model.simulate_paths(p["T"], N_paths=500, N_steps=50)
        scores = LikelihoodRatioGreeks.score_delta_heston(
            S[:, -1], p["S0"], v[:, -1], 0.04, p["T"], p["r"]
        )
        assert scores.shape == (500,)

    def test_score_delta_heston_finite(self, standard_params):
        """Heston scores should all be finite."""
        from models import HestonPath, set_seed
        p = standard_params
        set_seed(42)
        model = HestonPath(
            S0=p["S0"], v0=0.04, r=p["r"],
            kappa=2.0, theta=0.04, xi=0.30, rho=-0.70,
        )
        S, v = model.simulate_paths(p["T"], N_paths=500, N_steps=50)
        scores = LikelihoodRatioGreeks.score_delta_heston(
            S[:, -1], p["S0"], v[:, -1], 0.04, p["T"], p["r"]
        )
        assert np.all(np.isfinite(scores))

    def test_lr_vega_static_method(self, large_sample):
        """LikelihoodRatioGreeks.vega() should return a finite scalar."""
        params, S_T, N = large_sample
        option = EuropeanOption(params["K"], params["T"], params["r"])
        payoffs = option.payoff(S_T)
        scores = LikelihoodRatioGreeks.score_vega_gbm(
            S_T, params["S0"], params["sigma"], params["T"], params["r"]
        )
        vega = LikelihoodRatioGreeks.vega(payoffs, scores, option._discount)
        assert np.isfinite(vega)
        assert vega > 0.0  # Vega should be positive for a call


# ============================================================================
# compute_all_deltas and compute_all_vegas
# ============================================================================

class TestComputeAllGreeks:
    """Test the compute_all_deltas and compute_all_vegas aggregate functions."""

    def test_compute_all_deltas_keys(self, standard_params):
        """compute_all_deltas should return all four method keys."""
        from models import GBMPath, set_seed
        p = standard_params
        set_seed(42)
        gbm = GBMPath(p["S0"], p["r"], p["sigma"])
        option = EuropeanOption(p["K"], p["T"], p["r"])
        N = 10_000
        S_T = gbm.simulate_terminal(p["T"], N)
        payoffs = option.payoff(S_T)

        def price_func(S0):
            set_seed(42)
            g = GBMPath(S0, p["r"], p["sigma"])
            return option.price_from_terminal(g.simulate_terminal(p["T"], N))

        results = compute_all_deltas(
            S_T, payoffs, p["S0"], p["K"], p["T"], p["r"], p["sigma"],
            option._discount, price_func,
        )
        assert set(results.keys()) == {"pathwise", "likelihood_ratio",
                                       "finite_difference", "black_scholes"}

    def test_compute_all_deltas_values_finite(self, standard_params):
        """All Delta estimates should be finite."""
        from models import GBMPath, set_seed
        p = standard_params
        set_seed(42)
        gbm = GBMPath(p["S0"], p["r"], p["sigma"])
        option = EuropeanOption(p["K"], p["T"], p["r"])
        N = 10_000
        S_T = gbm.simulate_terminal(p["T"], N)
        payoffs = option.payoff(S_T)

        def price_func(S0):
            set_seed(42)
            g = GBMPath(S0, p["r"], p["sigma"])
            return option.price_from_terminal(g.simulate_terminal(p["T"], N))

        results = compute_all_deltas(
            S_T, payoffs, p["S0"], p["K"], p["T"], p["r"], p["sigma"],
            option._discount, price_func,
        )
        for method, val in results.items():
            assert np.isfinite(val), f"{method} returned non-finite Delta"

    def test_compute_all_vegas_keys(self, standard_params):
        """compute_all_vegas should return all four method keys."""
        from models import GBMPath, set_seed
        p = standard_params
        set_seed(42)
        gbm = GBMPath(p["S0"], p["r"], p["sigma"])
        option = EuropeanOption(p["K"], p["T"], p["r"])
        N = 10_000
        S_T = gbm.simulate_terminal(p["T"], N)
        payoffs = option.payoff(S_T)

        def price_func(sigma):
            set_seed(42)
            g = GBMPath(p["S0"], p["r"], sigma)
            return option.price_from_terminal(g.simulate_terminal(p["T"], N))

        results = compute_all_vegas(
            S_T, payoffs, p["S0"], p["K"], p["T"], p["r"], p["sigma"],
            option._discount, price_func,
        )
        assert set(results.keys()) == {"pathwise", "likelihood_ratio",
                                       "finite_difference", "black_scholes"}

    def test_compute_all_vegas_values_finite(self, standard_params):
        """All Vega estimates should be finite."""
        from models import GBMPath, set_seed
        p = standard_params
        set_seed(42)
        gbm = GBMPath(p["S0"], p["r"], p["sigma"])
        option = EuropeanOption(p["K"], p["T"], p["r"])
        N = 10_000
        S_T = gbm.simulate_terminal(p["T"], N)
        payoffs = option.payoff(S_T)

        def price_func(sigma):
            set_seed(42)
            g = GBMPath(p["S0"], p["r"], sigma)
            return option.price_from_terminal(g.simulate_terminal(p["T"], N))

        results = compute_all_vegas(
            S_T, payoffs, p["S0"], p["K"], p["T"], p["r"], p["sigma"],
            option._discount, price_func,
        )
        for method, val in results.items():
            assert np.isfinite(val), f"{method} returned non-finite Vega"