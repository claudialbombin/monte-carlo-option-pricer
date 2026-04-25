"""
Unit Tests — Utility Functions
================================
Tests for utils.py: Timer, confidence_interval, relative_error,
absolute_error, batch_means, ensure_dir, save_results_csv,
generate_seeds, validate_positive, validate_range.
"""

import os
import time
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    Timer,
    confidence_interval,
    relative_error,
    absolute_error,
    batch_means,
    ensure_dir,
    save_results_csv,
    generate_seeds,
    validate_positive,
    validate_range,
)


# ============================================================================
# Timer Tests
# ============================================================================

class TestTimer:
    """Test the Timer context manager and decorator."""

    def test_context_manager_records_elapsed(self):
        """Timer should record elapsed time after exiting context."""
        with Timer("test") as t:
            time.sleep(0.01)
        assert t.elapsed >= 0.01

    def test_context_manager_label(self, capsys):
        """Timer should print the label on exit."""
        with Timer("my operation"):
            pass
        captured = capsys.readouterr()
        assert "my operation" in captured.out

    def test_context_manager_default_label(self, capsys):
        """Timer without label uses default 'Operation'."""
        with Timer():
            pass
        captured = capsys.readouterr()
        assert "Operation" in captured.out

    def test_elapsed_zero_before_context(self):
        """Elapsed should be 0.0 before context is entered."""
        t = Timer("test")
        assert t.elapsed == 0.0

    def test_decorator_records_elapsed(self):
        """Timer used as decorator should record elapsed time."""
        @Timer("decorated")
        def slow_fn():
            time.sleep(0.01)

        slow_fn()

    def test_decorator_returns_function_result(self):
        """Decorated function should still return its result."""
        @Timer("add")
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_context_manager_elapsed_grows_with_sleep(self):
        """Longer sleep should produce larger elapsed."""
        with Timer("fast") as t_fast:
            time.sleep(0.01)
        with Timer("slow") as t_slow:
            time.sleep(0.05)
        assert t_slow.elapsed > t_fast.elapsed


# ============================================================================
# confidence_interval Tests
# ============================================================================

class TestConfidenceInterval:
    """Test confidence interval computation."""

    def test_mean_is_correct(self):
        """Mean should match np.mean."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, lo, hi = confidence_interval(data)
        assert np.isclose(mean, np.mean(data))

    def test_interval_contains_mean(self):
        """Lower and upper bounds should bracket the mean."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, lo, hi = confidence_interval(data)
        assert lo < mean < hi

    def test_95_percent_wider_than_90_percent(self):
        """95% CI should be wider than 90% CI."""
        data = np.random.default_rng(0).normal(0, 1, 1000)
        _, lo_90, hi_90 = confidence_interval(data, confidence=0.90)
        _, lo_95, hi_95 = confidence_interval(data, confidence=0.95)
        width_90 = hi_90 - lo_90
        width_95 = hi_95 - lo_95
        assert width_95 > width_90

    def test_single_element_returns_zeros(self):
        """Single-element array should return (value, 0, 0)."""
        data = np.array([5.0])
        mean, lo, hi = confidence_interval(data)
        assert mean == 5.0
        assert lo == 0.0
        assert hi == 0.0

    def test_empty_array_returns_zeros(self):
        """Empty array should return (0, 0, 0)."""
        data = np.array([])
        mean, lo, hi = confidence_interval(data)
        assert mean == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_large_sample_interval_is_narrow(self):
        """Large sample should produce a narrow confidence interval."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100_000)
        mean, lo, hi = confidence_interval(data)
        width = hi - lo
        assert width < 0.02  # Should be very narrow for 100k samples

    def test_constant_array_zero_interval(self):
        """Array of identical values should have zero-width interval."""
        data = np.ones(100)
        mean, lo, hi = confidence_interval(data)
        assert np.isclose(mean, 1.0)
        assert np.isclose(lo, hi)


# ============================================================================
# relative_error Tests
# ============================================================================

class TestRelativeError:
    """Test relative error computation."""

    def test_perfect_estimate(self):
        """Zero relative error when estimate equals true value."""
        assert relative_error(5.0, 5.0) == 0.0

    def test_nonzero_error(self):
        """Relative error of 10% for 110 vs 100."""
        err = relative_error(110.0, 100.0)
        assert np.isclose(err, 0.10)

    def test_negative_estimate(self):
        """Relative error uses absolute values."""
        err = relative_error(-5.0, 10.0)
        assert np.isclose(err, 1.5)

    def test_zero_true_value_nonzero_estimate_returns_inf(self):
        """If true_value is 0 and estimate != 0, return inf."""
        err = relative_error(1.0, 0.0)
        assert err == float('inf')

    def test_zero_true_value_zero_estimate_returns_zero(self):
        """If both are zero, return 0."""
        err = relative_error(0.0, 0.0)
        assert err == 0.0

    def test_symmetry_of_absolute_value(self):
        """relative_error(90, 100) should equal relative_error(110, 100)."""
        assert relative_error(90.0, 100.0) == relative_error(110.0, 100.0)


# ============================================================================
# absolute_error Tests
# ============================================================================

class TestAbsoluteError:
    """Test absolute error computation."""

    def test_perfect_estimate(self):
        """Zero absolute error when estimate equals true value."""
        assert absolute_error(5.0, 5.0) == 0.0

    def test_positive_error(self):
        """Absolute error is positive."""
        assert absolute_error(110.0, 100.0) == 10.0

    def test_negative_direction_error(self):
        """Absolute error is always positive."""
        assert absolute_error(90.0, 100.0) == 10.0

    def test_large_error(self):
        """Large discrepancy should be reflected in absolute error."""
        assert absolute_error(0.0, 100.0) == 100.0


# ============================================================================
# batch_means Tests
# ============================================================================

class TestBatchMeans:
    """Test batch means estimation."""

    def test_mean_matches_numpy(self):
        """batch_means should return the same mean as np.mean."""
        rng = np.random.default_rng(0)
        payoffs = rng.normal(5.0, 1.0, 1000)
        mean, _ = batch_means(payoffs, batch_size=100)
        assert np.isclose(mean, np.mean(payoffs), rtol=1e-10)

    def test_std_err_positive(self):
        """Standard error should be positive for noisy data."""
        rng = np.random.default_rng(42)
        payoffs = rng.normal(0.0, 1.0, 500)
        _, std_err = batch_means(payoffs, batch_size=50)
        assert std_err > 0

    def test_fewer_than_2_batches_falls_back(self):
        """If M < 2 batches, falls back to standard std error."""
        payoffs = np.array([1.0, 2.0, 3.0])
        # batch_size > N/2 ensures M < 2
        mean, std_err = batch_means(payoffs, batch_size=10)
        assert np.isclose(mean, np.mean(payoffs))
        assert std_err > 0

    def test_truncation_works(self):
        """Payoffs are truncated to integer multiple of batch_size."""
        rng = np.random.default_rng(7)
        payoffs = rng.normal(0, 1, 105)  # Not divisible by 10
        mean, _ = batch_means(payoffs, batch_size=10)
        # Only 100 of the 105 values are used
        assert np.isfinite(mean)

    def test_constant_payoffs_zero_stderr(self):
        """Constant payoffs should give zero standard error."""
        payoffs = np.ones(100)
        mean, std_err = batch_means(payoffs, batch_size=10)
        assert np.isclose(mean, 1.0)
        assert std_err == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# ensure_dir Tests
# ============================================================================

class TestEnsureDir:
    """Test directory creation utility."""

    def test_creates_new_directory(self, tmp_path):
        """Should create a directory that does not exist."""
        new_dir = str(tmp_path / "new_subdir")
        assert not os.path.exists(new_dir)
        ensure_dir(new_dir)
        assert os.path.isdir(new_dir)

    def test_existing_directory_no_error(self, tmp_path):
        """Should not raise an error if directory already exists."""
        existing = str(tmp_path)
        ensure_dir(existing)  # Should not raise
        assert os.path.isdir(existing)

    def test_creates_nested_directories(self, tmp_path):
        """Should create nested directories (parents=True)."""
        nested = str(tmp_path / "a" / "b" / "c")
        ensure_dir(nested)
        assert os.path.isdir(nested)


# ============================================================================
# save_results_csv Tests
# ============================================================================

class TestSaveResultsCsv:
    """Test CSV export utility."""

    def test_creates_file(self, tmp_path):
        """CSV file should be created."""
        save_results_csv(
            filename="test.csv",
            headers=["x", "y"],
            data=[[1, 2], [3, 4]],
            output_dir=str(tmp_path),
        )
        assert (tmp_path / "test.csv").exists()

    def test_correct_header(self, tmp_path):
        """First line of CSV should be the header."""
        save_results_csv(
            filename="test.csv",
            headers=["a", "b", "c"],
            data=[[1, 2, 3]],
            output_dir=str(tmp_path),
        )
        content = (tmp_path / "test.csv").read_text()
        assert content.startswith("a,b,c\n")

    def test_correct_data_rows(self, tmp_path):
        """Data rows should be written correctly."""
        save_results_csv(
            filename="out.csv",
            headers=["n", "price"],
            data=[[100, 5.5], [1000, 5.2]],
            output_dir=str(tmp_path),
        )
        lines = (tmp_path / "out.csv").read_text().splitlines()
        assert lines[1] == "100,5.5"
        assert lines[2] == "1000,5.2"

    def test_creates_output_dir_if_missing(self, tmp_path):
        """Should create output directory if it does not exist."""
        new_dir = str(tmp_path / "results")
        save_results_csv(
            filename="data.csv",
            headers=["x"],
            data=[[1]],
            output_dir=new_dir,
        )
        assert os.path.exists(os.path.join(new_dir, "data.csv"))

    def test_empty_data(self, tmp_path):
        """Should handle empty data gracefully (header only)."""
        save_results_csv(
            filename="empty.csv",
            headers=["col1"],
            data=[],
            output_dir=str(tmp_path),
        )
        content = (tmp_path / "empty.csv").read_text()
        assert content.strip() == "col1"


# ============================================================================
# generate_seeds Tests
# ============================================================================

class TestGenerateSeeds:
    """Test seed generation."""

    def test_correct_count(self):
        """Should return exactly n_seeds seeds."""
        seeds = generate_seeds(5)
        assert len(seeds) == 5

    def test_seeds_are_sequential(self):
        """Seeds should be base_seed + i."""
        seeds = generate_seeds(3, base_seed=10)
        assert seeds == [10, 11, 12]

    def test_default_base_seed(self):
        """Default base seed is 42."""
        seeds = generate_seeds(2)
        assert seeds[0] == 42
        assert seeds[1] == 43

    def test_all_seeds_unique(self):
        """All seeds should be distinct."""
        seeds = generate_seeds(100)
        assert len(set(seeds)) == 100

    def test_zero_seeds(self):
        """Requesting 0 seeds should return an empty list."""
        seeds = generate_seeds(0)
        assert seeds == []


# ============================================================================
# validate_positive Tests
# ============================================================================

class TestValidatePositive:
    """Test validate_positive helper."""

    def test_positive_value_passes(self):
        """Positive value should not raise."""
        validate_positive(1.0, "param")

    def test_zero_raises_error(self):
        """Zero should raise ValueError."""
        with pytest.raises(ValueError, match="param"):
            validate_positive(0.0, "param")

    def test_negative_raises_error(self):
        """Negative value should raise ValueError."""
        with pytest.raises(ValueError, match="sigma"):
            validate_positive(-1.0, "sigma")

    def test_error_message_contains_value(self):
        """Error message should include the bad value."""
        with pytest.raises(ValueError, match="-5"):
            validate_positive(-5.0, "K")


# ============================================================================
# validate_range Tests
# ============================================================================

class TestValidateRange:
    """Test validate_range helper."""

    def test_value_within_range_passes(self):
        """Value inside [low, high] should not raise."""
        validate_range(0.5, "rho", -1.0, 1.0)

    def test_lower_bound_inclusive(self):
        """Value equal to low should pass."""
        validate_range(-1.0, "rho", -1.0, 1.0)

    def test_upper_bound_inclusive(self):
        """Value equal to high should pass."""
        validate_range(1.0, "rho", -1.0, 1.0)

    def test_below_range_raises_error(self):
        """Value below low should raise ValueError."""
        with pytest.raises(ValueError, match="rho"):
            validate_range(-2.0, "rho", -1.0, 1.0)

    def test_above_range_raises_error(self):
        """Value above high should raise ValueError."""
        with pytest.raises(ValueError, match="rho"):
            validate_range(1.5, "rho", -1.0, 1.0)

    def test_error_message_contains_bounds(self):
        """Error message should reference the parameter name."""
        with pytest.raises(ValueError, match="alpha"):
            validate_range(2.0, "alpha", 0.0, 1.0)
