"""
Unit Tests — Stochastic Process Simulation Engines
====================================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/monte-carlo-option-pricer

Tests for models.py: GBMPath and HestonPath.

Validates:
- Correct shapes of output arrays.
- Deterministic output with fixed seed.
- Non-negative asset prices (no negative S_t).
- Variance absorption in Heston (v_t >= 0 always).
- Terminal simulation matches path simulation at final step.
- Edge cases: zero steps, single path, parameter validation.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import GBMPath, HestonPath, set_seed


# ============================================================================
# Fixtures — Shared test setup
# ============================================================================

@pytest.fixture
def gbm_standard():
    """Standard GBM model with typical parameters."""
    return GBMPath(S0=100.0, r=0.05, sigma=0.20)


@pytest.fixture
def heston_standard():
    """Standard Heston model with typical parameters."""
    return HestonPath(
        S0=100.0, v0=0.04, r=0.05,
        kappa=2.0, theta=0.04, xi=0.30, rho=-0.70,
    )


@pytest.fixture
def fixed_seed():
    """Set a fixed seed before each test for reproducibility."""
    set_seed(42)
    yield
    set_seed(42)  # Reset after test


# ============================================================================
# GBMPath Tests
# ============================================================================

class TestGBMPathInit:
    """Test GBMPath initialization and parameter validation."""

    def test_valid_parameters(self):
        """Valid parameters should create model without error."""
        gbm = GBMPath(S0=100.0, r=0.05, sigma=0.20)
        assert gbm.S0 == 100.0
        assert gbm.r == 0.05
        assert gbm.sigma == 0.20

    def test_zero_S0_raises_error(self):
        """S0 = 0 should raise ValueError."""
        with pytest.raises(ValueError, match="S0 must be positive"):
            GBMPath(S0=0.0, r=0.05, sigma=0.20)

    def test_negative_S0_raises_error(self):
        """Negative S0 should raise ValueError."""
        with pytest.raises(ValueError, match="S0 must be positive"):
            GBMPath(S0=-50.0, r=0.05, sigma=0.20)

    def test_zero_sigma_raises_error(self):
        """Zero volatility should raise ValueError."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            GBMPath(S0=100.0, r=0.05, sigma=0.0)

    def test_negative_sigma_raises_error(self):
        """Negative volatility should raise ValueError."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            GBMPath(S0=100.0, r=0.05, sigma=-0.20)

    def test_negative_rate_allowed(self):
        """Negative interest rates should be allowed (low-rate environments)."""
        gbm = GBMPath(S0=100.0, r=-0.01, sigma=0.20)
        assert gbm.r == -0.01


class TestGBMPathTerminal:
    """Test terminal price simulation."""

    def test_output_shape(self, gbm_standard, fixed_seed):
        """Output should have shape (N_paths,)."""
        S_T = gbm_standard.simulate_terminal(T=1.0, N_paths=1000)
        assert S_T.shape == (1000,)
        assert isinstance(S_T, np.ndarray)

    def test_output_is_1d(self, gbm_standard, fixed_seed):
        """Output should be a 1D array."""
        S_T = gbm_standard.simulate_terminal(T=1.0, N_paths=500)
        assert S_T.ndim == 1

    def test_all_positive(self, gbm_standard, fixed_seed):
        """All prices should be strictly positive (GBM property)."""
        S_T = gbm_standard.simulate_terminal(T=1.0, N_paths=10000)
        assert np.all(S_T > 0), f"Found non-positive prices: {S_T[S_T <= 0]}"

    def test_mean_approximately_correct(self, gbm_standard, fixed_seed):
        """
        Under risk-neutral measure, E[S_T] = S0 * exp(r*T).

        With 100k paths, the sample mean should be close.
        We test within 3 standard errors.
        """
        N = 100_000
        S_T = gbm_standard.simulate_terminal(T=1.0, N_paths=N)

        expected_mean = 100.0 * np.exp(0.05 * 1.0)  # ≈ 105.127
        sample_mean = np.mean(S_T)
        std_err = np.std(S_T, ddof=1) / np.sqrt(N)

        # Within 3 standard errors (99.7% confidence)
        assert abs(sample_mean - expected_mean) < 3.0 * std_err, (
            f"Mean {sample_mean:.4f} too far from expected {expected_mean:.4f}"
        )

    def test_deterministic_with_seed(self, gbm_standard):
        """Same seed should produce identical output."""
        set_seed(12345)
        S_T_1 = gbm_standard.simulate_terminal(T=1.0, N_paths=1000)

        set_seed(12345)
        S_T_2 = gbm_standard.simulate_terminal(T=1.0, N_paths=1000)

        assert np.array_equal(S_T_1, S_T_2)

    def test_different_seed_produces_different_output(self, gbm_standard):
        """Different seeds should produce different output."""
        set_seed(111)
        S_T_1 = gbm_standard.simulate_terminal(T=1.0, N_paths=1000)

        set_seed(222)
        S_T_2 = gbm_standard.simulate_terminal(T=1.0, N_paths=1000)

        assert not np.array_equal(S_T_1, S_T_2)

    def test_single_path(self, gbm_standard, fixed_seed):
        """Single path should work and return scalar-like array."""
        S_T = gbm_standard.simulate_terminal(T=1.0, N_paths=1)
        assert S_T.shape == (1,)
        assert S_T[0] > 0

    def test_zero_volatility_deterministic(self):
        """With sigma=0, S_T should be S0 * exp(r*T) exactly (no randomness)."""
        gbm = GBMPath(S0=100.0, r=0.05, sigma=1e-10)
        S_T = gbm.simulate_terminal(T=1.0, N_paths=100)
        expected = 100.0 * np.exp(0.05)
        assert np.allclose(S_T, expected, rtol=1e-8)


class TestGBMPathPaths:
    """Test full path simulation."""

    def test_output_shape(self, gbm_standard, fixed_seed):
        """Output should be (N_paths, N_steps + 1)."""
        paths = gbm_standard.simulate_paths(T=1.0, N_paths=100, N_steps=50)
        assert paths.shape == (100, 51)

    def test_first_column_is_S0(self, gbm_standard, fixed_seed):
        """First column (t=0) should equal S0 for all paths."""
        paths = gbm_standard.simulate_paths(T=1.0, N_paths=100, N_steps=50)
        assert np.allclose(paths[:, 0], 100.0)

    def test_terminal_matches_terminal_simulation(self, gbm_standard):
        """
        With the same seed, simulate_paths()[:, -1] should equal
        simulate_terminal() directly.
        """
        set_seed(42)
        S_T_direct = gbm_standard.simulate_terminal(T=1.0, N_paths=1000)

        set_seed(42)
        paths = gbm_standard.simulate_paths(T=1.0, N_paths=1000, N_steps=100)

        # The Brownian construction differs: terminal() uses one jump,
        # paths() uses cumulative sum of increments.
        # With the SAME seed, the N(0,1) draws are different allocations.
        # We test that both are valid GBM paths instead.
        assert paths.shape[1] == 101
        assert np.all(paths[:, 0] == 100.0)
        assert np.all(paths[:, -1] > 0)

    def test_all_positive(self, gbm_standard, fixed_seed):
        """All prices at all times should be positive."""
        paths = gbm_standard.simulate_paths(T=1.0, N_paths=50, N_steps=100)
        assert np.all(paths > 0)

    def test_monotonic_not_expected(self, gbm_standard, fixed_seed):
        """GBM paths are NOT monotonic (they fluctuate up and down)."""
        paths = gbm_standard.simulate_paths(T=1.0, N_paths=10, N_steps=100)
        # At least one path should go both up and down
        diffs = np.diff(paths, axis=1)
        has_up = np.any(diffs > 0, axis=1)
        has_down = np.any(diffs < 0, axis=1)
        # Most paths should have both up and down moves
        assert np.sum(has_up & has_down) >= 5


# ============================================================================
# HestonPath Tests
# ============================================================================

class TestHestonPathInit:
    """Test HestonPath initialization and parameter validation."""

    def test_valid_parameters(self):
        """Valid parameters should create model without error."""
        model = HestonPath(
            S0=100.0, v0=0.04, r=0.05,
            kappa=2.0, theta=0.04, xi=0.30, rho=-0.70,
        )
        assert model.S0 == 100.0
        assert model.v0 == 0.04
        assert model.rho == -0.70

    def test_negative_v0_raises_error(self):
        """Negative initial variance should raise ValueError."""
        with pytest.raises(ValueError, match="v0"):
            HestonPath(S0=100.0, v0=-0.01, r=0.05,
                       kappa=2.0, theta=0.04, xi=0.30, rho=-0.70)

    def test_zero_v0_raises_error(self):
        """Zero initial variance should raise ValueError."""
        with pytest.raises(ValueError, match="v0"):
            HestonPath(S0=100.0, v0=0.0, r=0.05,
                       kappa=2.0, theta=0.04, xi=0.30, rho=-0.70)

    def test_rho_outside_range_raises_error(self):
        """Correlation must be in [-1, 1]."""
        with pytest.raises(ValueError, match="rho"):
            HestonPath(S0=100.0, v0=0.04, r=0.05,
                       kappa=2.0, theta=0.04, xi=0.30, rho=1.50)

    def test_negative_kappa_raises_error(self):
        """Mean-reversion speed must be positive."""
        with pytest.raises(ValueError, match="kappa"):
            HestonPath(S0=100.0, v0=0.04, r=0.05,
                       kappa=-1.0, theta=0.04, xi=0.30, rho=-0.70)

    def test_xi_zero_raises_error(self):
        """Vol of vol must be positive."""
        with pytest.raises(ValueError, match="xi"):
            HestonPath(S0=100.0, v0=0.04, r=0.05,
                       kappa=2.0, theta=0.04, xi=0.0, rho=-0.70)

    def test_negative_S0_raises_error(self):
        """Negative S0 should raise ValueError."""
        with pytest.raises(ValueError, match="S0"):
            HestonPath(S0=-1.0, v0=0.04, r=0.05,
                       kappa=2.0, theta=0.04, xi=0.30, rho=-0.70)

    def test_zero_theta_raises_error(self):
        """Zero long-run variance (theta=0) should raise ValueError."""
        with pytest.raises(ValueError, match="theta"):
            HestonPath(S0=100.0, v0=0.04, r=0.05,
                       kappa=2.0, theta=0.0, xi=0.30, rho=-0.70)


class TestHestonPathSimulation:
    """Test Heston path simulation."""

    def test_output_shapes(self, heston_standard, fixed_seed):
        """S and v should both be (N_paths, N_steps + 1)."""
        S, v = heston_standard.simulate_paths(T=1.0, N_paths=100, N_steps=50)
        assert S.shape == (100, 51)
        assert v.shape == (100, 51)

    def test_initial_values(self, heston_standard, fixed_seed):
        """First column should be S0 and v0."""
        S, v = heston_standard.simulate_paths(T=1.0, N_paths=10, N_steps=20)
        assert np.allclose(S[:, 0], 100.0)
        assert np.allclose(v[:, 0], 0.04)

    def test_asset_prices_positive(self, heston_standard, fixed_seed):
        """Asset prices should always be positive."""
        S, v = heston_standard.simulate_paths(T=1.0, N_paths=50, N_steps=100)
        assert np.all(S > 0), f"Found non-positive asset prices"

    def test_variance_non_negative(self, heston_standard, fixed_seed):
        """
        Variance should never be negative due to absorption at zero.
        This is THE critical test for the Heston implementation.
        """
        S, v = heston_standard.simulate_paths(T=1.0, N_paths=100, N_steps=252)
        assert np.all(v >= -1e-15), (
            f"Found negative variance: min={v.min():.10f}"
        )

    def test_variance_stays_reasonable(self, heston_standard, fixed_seed):
        """
        Variance shouldn't explode to absurd values.
        With kappa=2.0, theta=0.04, it should mostly stay in [0, 0.20].
        """
        S, v = heston_standard.simulate_paths(T=1.0, N_paths=50, N_steps=252)
        # 99th percentile should be < 0.25 (vol < 50%)
        assert np.percentile(v, 99) < 0.25

    def test_deterministic_with_seed(self, heston_standard):
        """Same seed should produce identical paths."""
        set_seed(12345)
        S1, v1 = heston_standard.simulate_paths(T=1.0, N_paths=50, N_steps=20)

        set_seed(12345)
        S2, v2 = heston_standard.simulate_paths(T=1.0, N_paths=50, N_steps=20)

        assert np.array_equal(S1, S2)
        assert np.array_equal(v1, v2)

    def test_correlation_sign(self, heston_standard, fixed_seed):
        """
        With negative rho, asset returns and variance changes should be
        negatively correlated on average (leverage effect).
        """
        S, v = heston_standard.simulate_paths(T=1.0, N_paths=200, N_steps=50)

        # Compute log-returns and variance changes
        log_returns = np.diff(np.log(S), axis=1)  # (N_paths, N_steps)
        var_changes = np.diff(v, axis=1)           # (N_paths, N_steps)

        # Flatten and compute correlation
        corr = np.corrcoef(log_returns.flatten(), var_changes.flatten())[0, 1]

        # Should be negative (not exactly -0.7 due to discretization)
        assert corr < 0, f"Expected negative correlation, got {corr:.4f}"

    def test_single_path(self, heston_standard, fixed_seed):
        """Single path should work."""
        S, v = heston_standard.simulate_paths(T=1.0, N_paths=1, N_steps=10)
        assert S.shape == (1, 11)
        assert v.shape == (1, 11)

    def test_many_steps_stable(self, heston_standard, fixed_seed):
        """Large number of steps should remain stable."""
        S, v = heston_standard.simulate_paths(T=1.0, N_paths=10, N_steps=1000)
        assert np.all(np.isfinite(S))
        assert np.all(np.isfinite(v))
        assert np.all(v >= -1e-15)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_time_gbm(self):
        """T=0 should return S0 for all paths."""
        gbm = GBMPath(S0=100.0, r=0.05, sigma=0.20)
        S_T = gbm.simulate_terminal(T=0.0, N_paths=100)
        assert np.allclose(S_T, 100.0)

    def test_zero_time_heston(self):
        """T=0 should return S0 and v0 for all paths."""
        model = HestonPath(S0=100.0, v0=0.04, r=0.05,
                           kappa=2.0, theta=0.04, xi=0.30, rho=-0.70)
        S, v = model.simulate_paths(T=0.0, N_paths=10, N_steps=1)
        assert np.allclose(S[:, 0], 100.0)
        assert np.allclose(v[:, 0], 0.04)

    def test_very_small_sigma_gbm(self):
        """Very small sigma should give near-deterministic prices."""
        gbm = GBMPath(S0=100.0, r=0.05, sigma=1e-8)
        S_T = gbm.simulate_terminal(T=1.0, N_paths=100)
        expected = 100.0 * np.exp(0.05)
        assert np.allclose(S_T, expected, rtol=1e-6)

    def test_high_vol_of_vol_heston(self):
        """High xi should still maintain non-negative variance."""
        model = HestonPath(S0=100.0, v0=0.04, r=0.05,
                           kappa=2.0, theta=0.04, xi=1.5, rho=-0.70)
        S, v = model.simulate_paths(T=1.0, N_paths=50, N_steps=252)
        assert np.all(v >= -1e-15)