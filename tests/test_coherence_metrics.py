"""Tests for ISPC and CDPGC coherence metrics: shapes, ranges, edge cases."""

import numpy as np
import pytest

from src.phase_extraction import PhaseExtractor
from src.coherence_metrics import compute_ispc, compute_cdpgc, compute_ispc_map

NLEVELS = 3


@pytest.fixture
def extractor():
    return PhaseExtractor(nlevels=NLEVELS)


@pytest.fixture
def small_gray():
    np.random.seed(77)
    return np.random.rand(128, 128).astype(np.float64)


@pytest.fixture
def phase_maps_from_random(extractor, small_gray):
    """Phase maps extracted from a random 128x128 image."""
    return extractor.get_phase_maps(small_gray)


@pytest.fixture
def constant_phase_maps():
    """Phase maps where all phases are zero (perfect coherence).

    Mimics the shape structure of a 3-level DT-CWT on a 128x128 image.
    """
    return [
        np.zeros((64, 64, 6), dtype=np.float64),
        np.zeros((32, 32, 6), dtype=np.float64),
        np.zeros((16, 16, 6), dtype=np.float64),
    ]


@pytest.fixture
def random_uniform_phase_maps():
    """Phase maps drawn uniformly from [-pi, pi] (no coherence).

    Uses matching spatial dimensions for a 3-level decomposition.
    """
    rng = np.random.RandomState(123)
    return [
        rng.uniform(-np.pi, np.pi, (64, 64, 6)),
        rng.uniform(-np.pi, np.pi, (32, 32, 6)),
        rng.uniform(-np.pi, np.pi, (16, 16, 6)),
    ]


class TestComputeISPC:
    """Tests for compute_ispc."""

    def test_is01_output_shape(self, phase_maps_from_random):
        """IS-01: Output is a numpy array with 6 direction values."""
        result = compute_ispc(phase_maps_from_random)
        assert isinstance(result, np.ndarray)
        # Must contain one value per DT-CWT direction (6 orientations)
        assert 6 in result.shape, (
            f"Expected 6 (directions) in output shape, got {result.shape}"
        )

    def test_is02_values_in_unit_range(self, phase_maps_from_random):
        """IS-02: ISPC values (circular variance) lie in [0, 1]."""
        result = compute_ispc(phase_maps_from_random)
        assert result.min() >= -1e-10, "ISPC values should be >= 0"
        assert result.max() <= 1.0 + 1e-10, "ISPC values should be <= 1"

    def test_is03_constant_phase_gives_low_ispc(self, constant_phase_maps):
        """IS-03: Zero phase everywhere yields ISPC near 0 (high coherence)."""
        result = compute_ispc(constant_phase_maps)
        # Circular variance of identical phases is 0
        assert np.all(result < 0.05), (
            f"Constant phase should give near-zero ISPC, got max={result.max():.4f}"
        )

    def test_is04_random_phase_gives_high_ispc(self, random_uniform_phase_maps):
        """IS-04: Uniform random phase yields ISPC closer to 1 (low coherence)."""
        result = compute_ispc(random_uniform_phase_maps)
        # With enough spatial samples, circular variance of random phases
        # should be noticeably above zero.
        mean_ispc = np.mean(result)
        assert mean_ispc > 0.3, (
            f"Random phase should give elevated ISPC, got mean={mean_ispc:.4f}"
        )


class TestComputeISPCMap:
    """Tests for compute_ispc_map."""

    def test_im01_returns_2d_array(self, phase_maps_from_random):
        """IM-01: compute_ispc_map returns a 2D numpy array."""
        result = compute_ispc_map(phase_maps_from_random, patch_size=16)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_im02_values_in_unit_range(self, phase_maps_from_random):
        """IM-02: ISPC map values lie in [0, 1]."""
        result = compute_ispc_map(phase_maps_from_random, patch_size=16)
        assert result.min() >= -1e-10, "ISPC map values should be >= 0"
        assert result.max() <= 1.0 + 1e-10, "ISPC map values should be <= 1"

    def test_im_raises_on_single_level(self):
        """compute_ispc_map raises ValueError if fewer than 2 levels given."""
        single_level = [np.zeros((32, 32, 6))]
        with pytest.raises(ValueError, match="at least 2 levels"):
            compute_ispc_map(single_level)


class TestComputeCDPGC:
    """Tests for compute_cdpgc."""

    def test_cd01_output_shape(self, phase_maps_from_random):
        """CD-01: Output shape is (n_levels,)."""
        result = compute_cdpgc(phase_maps_from_random)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(phase_maps_from_random),)

    def test_cd02_values_non_negative(self, phase_maps_from_random):
        """CD-02: CDPGC values (variance) are non-negative."""
        result = compute_cdpgc(phase_maps_from_random)
        assert np.all(result >= -1e-15), "Variance-based CDPGC must be >= 0"

    def test_cd03_constant_phase_gives_zero_cdpgc(self, constant_phase_maps):
        """CD-03: Constant (zero) phase maps yield CDPGC = 0."""
        result = compute_cdpgc(constant_phase_maps)
        np.testing.assert_allclose(
            result, 0.0, atol=1e-10,
            err_msg="Constant phase should give zero CDPGC",
        )
