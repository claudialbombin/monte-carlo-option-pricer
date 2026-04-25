"""
Unit Tests — Option Payoff Classes
====================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/monte-carlo-option-pricer

Tests for options.py: EuropeanOption, AsianOption, BarrierOption.

Validates:
- Correct payoff computation for simple cases.
- Discount factor application.
- Barrier knockout logic.
- Edge cases: at-the-money, deep OTM, deep ITM, barrier at S0.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from options import EuropeanOption, AsianOption, BarrierOption


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def european_call():
    """Standard European call: K=100, T=1, r=5%."""
    return EuropeanOption(K=100.0, T=1.0, r=0.05)


@pytest.fixture
def asian_call():
    """Standard Asian call: K=100, T=1, r=5%."""
    return AsianOption(K=100.0, T=1.0, r=0.05)


@pytest.fixture
def barrier_call():
    """Standard barrier call: K=100, B=120, T=1, r=5%."""
    return BarrierOption(K=100.0, B=120.0, T=1.0, r=0.05)


# ============================================================================
# EuropeanOption Tests
# ============================================================================

class TestEuropeanOptionInit:
    """Test EuropeanOption initialization."""

    def test_valid_parameters(self):
        """Valid parameters should create option."""
        option = EuropeanOption(K=100.0, T=1.0, r=0.05)
        assert option.K == 100.0
        assert option.T == 1.0

    def test_negative_strike_raises_error(self):
        """Negative strike should raise ValueError."""
        with pytest.raises(ValueError):
            EuropeanOption(K=-50.0, T=1.0, r=0.05)

    def test_zero_maturity_raises_error(self):
        """Zero T should raise ValueError."""
        with pytest.raises(ValueError):
            EuropeanOption(K=100.0, T=0.0, r=0.05)


class TestEuropeanOptionPayoff:
    """Test European option payoff computation."""

    def test_all_out_of_the_money(self, european_call):
        """If all S_T < K, all payoffs should be zero."""
        S_T = np.array([80.0, 90.0, 95.0, 99.0])
        payoffs = european_call.payoff(S_T)
        assert np.all(payoffs == 0.0)

    def test_all_in_the_money(self, european_call):
        """If all S_T > K, payoffs should be S_T - K."""
        S_T = np.array([105.0, 110.0, 120.0, 150.0])
        payoffs = european_call.payoff(S_T)
        expected = np.array([5.0, 10.0, 20.0, 50.0])
        assert np.allclose(payoffs, expected)

    def test_mixed_moneyness(self, european_call):
        """Mixed in/out of the money."""
        S_T = np.array([80.0, 105.0, 90.0, 110.0])
        payoffs = european_call.payoff(S_T)
        expected = np.array([0.0, 5.0, 0.0, 10.0])
        assert np.allclose(payoffs, expected)

    def test_exactly_at_the_money(self, european_call):
        """S_T = K should give zero payoff."""
        S_T = np.array([100.0, 100.0])
        payoffs = european_call.payoff(S_T)
        assert np.all(payoffs == 0.0)

    def test_single_value(self, european_call):
        """Single path should return array of shape (1,)."""
        S_T = np.array([105.0])
        payoffs = european_call.payoff(S_T)
        assert payoffs.shape == (1,)
        assert payoffs[0] == 5.0


class TestEuropeanOptionPrice:
    """Test European option pricing (discounting)."""

    def test_price_is_discounted(self, european_call):
        """Price should be less than average raw payoff due to discounting."""
        S_T = np.array([110.0] * 100)  # All deep ITM
        price = european_call.price_from_terminal(S_T)
        raw_mean = np.mean(european_call.payoff(S_T))  # = 10.0
        expected_price = raw_mean * np.exp(-0.05)
        assert np.isclose(price, expected_price)

    def test_price_zero_for_deep_otm(self, european_call):
        """Deep OTM options should have near-zero price."""
        S_T = np.array([1.0] * 10000)
        price = european_call.price_from_terminal(S_T)
        assert price == 0.0

    def test_price_from_paths_same_as_terminal(self, european_call):
        """price_from_paths should match price_from_terminal for same S_T."""
        paths = np.column_stack([
            np.array([100.0, 100.0, 100.0]),  # t0
            np.array([105.0, 110.0, 95.0]),   # t1
            np.array([108.0, 115.0, 98.0]),   # t2
        ]).T  # (3, 3)

        price_paths = european_call.price_from_paths(paths)
        price_terminal = european_call.price_from_terminal(paths[:, -1])
        assert np.isclose(price_paths, price_terminal)


# ============================================================================
# AsianOption Tests
# ============================================================================

class TestAsianOptionInit:
    """Test AsianOption initialization."""

    def test_valid_parameters(self):
        """Valid parameters should create option."""
        option = AsianOption(K=100.0, T=1.0, r=0.05)

    def test_invalid_strike_raises_error(self):
        """Negative strike should raise ValueError."""
        with pytest.raises(ValueError):
            AsianOption(K=-10.0, T=1.0, r=0.05)

    def test_zero_maturity_raises_error(self):
        """T=0 should raise ValueError."""
        with pytest.raises(ValueError):
            AsianOption(K=100.0, T=0.0, r=0.05)


class TestAsianOptionPayoff:
    """Test Asian option payoff computation."""

    def setup_method(self):
        """Create test paths."""
        # 3 paths, 4 time steps (t0, t1, t2, t3)
        # Average excludes t0 (S_0)
        self.paths = np.array([
            [100.0, 105.0, 110.0, 115.0],  # avg = 110, payoff = 10
            [100.0, 95.0,  90.0,  85.0],   # avg = 90,  payoff = 0
            [100.0, 100.0, 100.0, 100.0],  # avg = 100, payoff = 0
        ])

    def test_payoff_above_strike(self, asian_call):
        """Path with average > K should have positive payoff."""
        payoffs = asian_call.payoff(self.paths)
        assert payoffs[0] == pytest.approx(10.0)

    def test_payoff_below_strike(self, asian_call):
        """Path with average < K should have zero payoff."""
        payoffs = asian_call.payoff(self.paths)
        assert payoffs[1] == 0.0

    def test_payoff_at_strike(self, asian_call):
        """Path with average = K should have zero payoff."""
        payoffs = asian_call.payoff(self.paths)
        assert payoffs[2] == 0.0

    def test_payoff_shape(self, asian_call):
        """Payoff should be 1D array with length = N_paths."""
        payoffs = asian_call.payoff(self.paths)
        assert payoffs.shape == (3,)

    def test_excludes_initial_price(self, asian_call):
        """
        Average should exclude S_0 (column 0).
        Path: [S0=100, 120, 120, 120] -> avg(120,120,120) = 120, not 115.
        """
        paths = np.array([[100.0, 120.0, 120.0, 120.0]])
        payoffs = asian_call.payoff(paths)
        assert payoffs[0] == pytest.approx(20.0)


class TestAsianOptionPrice:
    """Test Asian option pricing."""

    def test_price_is_discounted(self, asian_call):
        """Price should incorporate discounting."""
        paths = np.array([
            [100.0, 120.0, 120.0, 120.0],  # avg = 120
        ])
        price = asian_call.price(paths)
        expected_avg = 120.0
        expected_payoff = expected_avg - 100.0  # = 20
        expected_price = expected_payoff * np.exp(-0.05)
        assert np.isclose(price, expected_price)

    def test_asian_cheaper_than_european(self, asian_call):
        """
        For paths with the same terminal value but different averages,
        the Asian option should be cheaper (averaging reduces volatility).
        """
        # Path: starts at 100, spikes to 200, ends at 100
        # Asian average is much lower than the terminal spike
        paths = np.array([
            [100.0, 200.0, 200.0, 200.0, 100.0],  # avg = 160 (5 cols, excl S0)
        ])

        asian_price = asian_call.price(paths)
        european = EuropeanOption(K=100.0, T=1.0, r=0.05)
        eu_price = european.price_from_terminal(paths[:, -1])

        # European should be zero (S_T = 100 = K), Asian should be > 0
        # Actually both would be zero if S_T=100. Let me fix the example.
        # Better: S_T = 150, but average lower
        paths2 = np.array([
            [100.0, 100.0, 100.0, 100.0, 150.0],  # avg = 112.5
        ])

        asian_price2 = asian_call.price(paths2)
        eu_price2 = european.price_from_terminal(paths2[:, -1])

        # European payoff = max(150-100, 0) = 50
        # Asian payoff = max(112.5-100, 0) = 12.5
        assert asian_price2 < eu_price2


# ============================================================================
# BarrierOption Tests
# ============================================================================

class TestBarrierOptionInit:
    """Test BarrierOption initialization."""

    def test_valid_parameters(self):
        """Valid parameters."""
        option = BarrierOption(K=100.0, B=120.0, T=1.0, r=0.05)
        assert option.B == 120.0

    def test_invalid_barrier_raises_error(self):
        """Negative barrier should raise ValueError."""
        with pytest.raises(ValueError):
            BarrierOption(K=100.0, B=-10.0, T=1.0, r=0.05)

    def test_zero_strike_raises_error(self):
        """K=0 should raise ValueError."""
        with pytest.raises(ValueError):
            BarrierOption(K=0.0, B=120.0, T=1.0, r=0.05)

    def test_zero_maturity_raises_error(self):
        """T=0 should raise ValueError."""
        with pytest.raises(ValueError):
            BarrierOption(K=100.0, B=120.0, T=0.0, r=0.05)


class TestBarrierOptionPayoff:
    """Test barrier option payoff with knockout logic."""

    def setup_method(self):
        """Create test paths."""
        # Path 0: stays below barrier, ends ITM -> payoff = 10
        # Path 1: touches barrier at step 2 -> knocked out -> payoff = 0
        # Path 2: stays below barrier, ends OTM -> payoff = 0
        # Path 3: touches barrier exactly -> knocked out -> payoff = 0
        self.paths = np.array([
            [100.0, 105.0, 110.0, 115.0],  # max=115 < 120, S_T=115 > 100
            [100.0, 115.0, 125.0, 110.0],  # max=125 >= 120, KNOCKED OUT
            [100.0, 95.0,  90.0,  85.0],   # max=100 < 120, S_T=85 < 100
            [100.0, 110.0, 120.0, 110.0],  # max=120 >= 120, KNOCKED OUT
        ])

    def test_surviving_path_payoff(self, barrier_call):
        """Path that doesn't hit barrier should get European payoff."""
        payoffs = barrier_call.payoff(self.paths)
        assert payoffs[0] == pytest.approx(15.0)  # 115 - 100

    def test_knocked_out_path_payoff(self, barrier_call):
        """Path that hits barrier should get zero payoff."""
        payoffs = barrier_call.payoff(self.paths)
        assert payoffs[1] == 0.0

    def test_otm_surviving_path(self, barrier_call):
        """Path that survives but is OTM should get zero."""
        payoffs = barrier_call.payoff(self.paths)
        assert payoffs[2] == 0.0

    def test_exact_barrier_knockout(self, barrier_call):
        """Path that exactly touches the barrier should be knocked out."""
        payoffs = barrier_call.payoff(self.paths)
        assert payoffs[3] == 0.0

    def test_payoff_shape(self, barrier_call):
        """Payoff should be 1D array with length = N_paths."""
        payoffs = barrier_call.payoff(self.paths)
        assert payoffs.shape == (4,)


class TestBarrierOptionKnockout:
    """Test the knockout probability estimation."""

    def test_knockout_probability(self, barrier_call):
        """Knockout probability should be correctly computed."""
        paths = np.array([
            [100.0, 110.0, 115.0],  # survives
            [100.0, 125.0, 110.0],  # knocked out
            [100.0, 105.0, 110.0],  # survives
            [100.0, 120.0, 105.0],  # knocked out
        ])
        price, prob = barrier_call.price_with_knockout_info(paths)
        assert prob == 0.5  # 2 out of 4 knocked out

    def test_barrier_below_S0_immediate_knockout(self):
        """If B <= S0, all paths should be immediately knocked out."""
        option = BarrierOption(K=100.0, B=90.0, T=1.0, r=0.05)
        paths = np.array([
            [100.0, 105.0, 110.0],
            [100.0, 95.0, 100.0],
        ])
        payoffs = option.payoff(paths)
        assert np.all(payoffs == 0.0)


# ============================================================================
# Discount Factor Tests
# ============================================================================

class TestDiscountFactor:
    """Test that discounting is correctly applied."""

    def test_european_discount(self):
        """Higher rates -> lower present value."""
        S_T = np.array([120.0])

        opt_r5 = EuropeanOption(K=100.0, T=1.0, r=0.05)
        opt_r10 = EuropeanOption(K=100.0, T=1.0, r=0.10)

        price_r5 = opt_r5.price_from_terminal(S_T)
        price_r10 = opt_r10.price_from_terminal(S_T)

        assert price_r10 < price_r5  # Higher rate = lower PV

    def test_zero_rate_no_discount(self):
        """With r=0, price = expected payoff."""
        option = EuropeanOption(K=100.0, T=1.0, r=0.0)
        S_T = np.array([80.0, 120.0])
        price = option.price_from_terminal(S_T)
        expected = np.mean([0.0, 20.0])
        assert np.isclose(price, expected)

    def test_very_long_maturity(self):
        """Very long maturity with positive rate -> near-zero price."""
        option = EuropeanOption(K=100.0, T=100.0, r=0.05)
        S_T = np.array([200.0])
        price = option.price_from_terminal(S_T)
        # Price should be heavily discounted
        assert price < 5.0  # 100 payoff discounted at 5% for 100 years


# ============================================================================
# BarrierOption.price and knockout_probability_estimate Tests
# ============================================================================

class TestBarrierOptionPrice:
    """Test BarrierOption.price() and knockout_probability_estimate."""

    def test_price_returns_scalar(self, barrier_call):
        """price() should return a float."""
        paths = np.array([
            [100.0, 105.0, 110.0, 115.0],  # survives, ITM
            [100.0, 95.0,  90.0,  85.0],   # survives, OTM
        ])
        p = barrier_call.price(paths)
        assert isinstance(p, float)

    def test_price_is_discounted(self, barrier_call):
        """price() should apply exp(-rT) discount."""
        paths = np.array([[100.0, 105.0, 110.0, 115.0]])  # survives, S_T=115
        p = barrier_call.price(paths)
        expected = (115.0 - 100.0) * np.exp(-0.05)
        assert np.isclose(p, expected)

    def test_price_all_knocked_out_is_zero(self, barrier_call):
        """If all paths are knocked out, price should be zero."""
        paths = np.array([
            [100.0, 125.0, 130.0, 140.0],
            [100.0, 121.0, 130.0, 115.0],
        ])
        assert barrier_call.price(paths) == 0.0

    def test_knockout_probability_estimate_none_before_call(self):
        """knockout_probability_estimate should be None before pricing."""
        option = BarrierOption(K=100.0, B=120.0, T=1.0, r=0.05)
        assert option.knockout_probability_estimate is None

    def test_knockout_probability_estimate_after_call(self, barrier_call):
        """knockout_probability_estimate should be set after price_with_knockout_info."""
        paths = np.array([
            [100.0, 110.0, 115.0],  # survives
            [100.0, 125.0, 110.0],  # knocked out
        ])
        barrier_call.price_with_knockout_info(paths)
        prob = barrier_call.knockout_probability_estimate
        assert prob == pytest.approx(0.5)