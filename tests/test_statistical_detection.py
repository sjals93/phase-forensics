"""Tests for GGDFitter and NeymanPearsonDetector: fitting, keys, score contracts."""

import numpy as np
import pytest

from src.statistical_detection import GGDFitter, NeymanPearsonDetector


@pytest.fixture
def fitter():
    return GGDFitter()


@pytest.fixture
def gaussian_data():
    """Standard normal samples (GGD beta=2)."""
    rng = np.random.RandomState(42)
    return rng.normal(0, 1, size=5000)


@pytest.fixture
def laplacian_data():
    """Laplacian samples (GGD beta=1)."""
    rng = np.random.RandomState(42)
    return rng.laplace(0, 1, size=5000)


@pytest.fixture
def real_stats():
    """Minimal real_stats dict for NeymanPearsonDetector."""
    return {
        'ispc_mean': 0.3,
        'ispc_std': 0.05,
        'cdpgc_mean': 0.01,
        'cdpgc_std': 0.005,
        'ispc_ggd': {'alpha': 0.05, 'beta': 2.0, 'mean': 0.3, 'std': 0.05},
        'cdpgc_ggd': {'alpha': 0.005, 'beta': 1.5, 'mean': 0.01, 'std': 0.005},
    }


class TestGGDFitter:
    """Tests for GGDFitter."""

    def test_gf01_gaussian_beta_near_2(self, fitter, gaussian_data):
        """GF-01: Fitting Gaussian data should yield beta near 2."""
        result = fitter.fit(gaussian_data)
        assert abs(result['beta'] - 2.0) < 0.5, (
            f"Expected beta near 2 for Gaussian, got {result['beta']:.3f}"
        )

    def test_gf02_laplacian_beta_near_1(self, fitter, laplacian_data):
        """GF-02: Fitting Laplacian data should yield beta near 1."""
        result = fitter.fit(laplacian_data)
        assert abs(result['beta'] - 1.0) < 0.5, (
            f"Expected beta near 1 for Laplacian, got {result['beta']:.3f}"
        )

    def test_gf03_output_has_required_keys(self, fitter, gaussian_data):
        """GF-03: Output dict contains alpha, beta, mean, std."""
        result = fitter.fit(gaussian_data)
        for key in ('alpha', 'beta', 'mean', 'std'):
            assert key in result, f"Missing key: {key}"

    def test_gf04_alpha_and_beta_positive(self, fitter, gaussian_data):
        """GF-04: alpha > 0 and beta > 0."""
        result = fitter.fit(gaussian_data)
        assert result['alpha'] > 0, f"alpha must be positive, got {result['alpha']}"
        assert result['beta'] > 0, f"beta must be positive, got {result['beta']}"

    def test_gf_handles_constant_data(self, fitter):
        """GGDFitter does not crash on constant (zero-variance) data."""
        data = np.ones(100)
        result = fitter.fit(data)
        assert 'alpha' in result
        assert 'beta' in result

    def test_gf_handles_small_data(self, fitter):
        """GGDFitter does not crash on very small arrays."""
        data = np.array([0.1, 0.2, 0.3])
        result = fitter.fit(data)
        assert result['alpha'] > 0
        assert result['beta'] > 0


class TestNeymanPearsonDetector:
    """Tests for NeymanPearsonDetector."""

    def test_np01_score_returns_float(self, real_stats):
        """NP-01: score() returns a Python float."""
        detector = NeymanPearsonDetector(real_stats)
        s = detector.score(0.3, 0.01)
        assert isinstance(s, float)

    def test_np02_score_near_zero_for_real_like_values(self, real_stats):
        """NP-02: Score near 0 when input matches the real distribution mean."""
        detector = NeymanPearsonDetector(real_stats)
        s = detector.score(
            real_stats['ispc_mean'],
            real_stats['cdpgc_mean'],
        )
        assert abs(s) < 1e-6, (
            f"Score at real mean should be ~0, got {s:.6f}"
        )

    def test_np03_anomalous_values_give_positive_score(self, real_stats):
        """NP-03: Values significantly above the real mean yield positive score."""
        detector = NeymanPearsonDetector(real_stats)
        # Use values far above the real mean (many sigma away)
        s = detector.score(
            real_stats['ispc_mean'] + 10 * real_stats['ispc_std'],
            real_stats['cdpgc_mean'] + 10 * real_stats['cdpgc_std'],
        )
        assert s > 0, f"Anomalous values should give positive score, got {s:.6f}"

    def test_np_score_monotonic_in_ispc(self, real_stats):
        """Higher ISPC (more anomalous) should yield higher score."""
        detector = NeymanPearsonDetector(real_stats)
        cdpgc_fixed = real_stats['cdpgc_mean']
        s_low = detector.score(real_stats['ispc_mean'], cdpgc_fixed)
        s_high = detector.score(real_stats['ispc_mean'] + 5 * real_stats['ispc_std'], cdpgc_fixed)
        assert s_high > s_low, "Score should increase with higher ISPC"
