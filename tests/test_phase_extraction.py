"""Tests for PhaseExtractor: output shapes, value ranges, determinism."""

import numpy as np
import pytest

from src.phase_extraction import PhaseExtractor

NLEVELS = 3


@pytest.fixture
def extractor():
    return PhaseExtractor(nlevels=NLEVELS)


@pytest.fixture
def small_gray():
    np.random.seed(99)
    return np.random.rand(128, 128).astype(np.float64)


@pytest.fixture
def small_rgb():
    np.random.seed(99)
    return np.random.rand(128, 128, 3).astype(np.float64)


class TestExtract:
    """Tests for PhaseExtractor.extract()."""

    def test_pe01_returns_correct_number_of_levels(self, extractor, small_gray):
        """PE-01: extract() returns a list with exactly nlevels entries."""
        coeffs = extractor.extract(small_gray)
        assert len(coeffs) == NLEVELS

    def test_pe02_each_level_shape(self, extractor, small_gray):
        """PE-02: Each level has shape (H_l, W_l, 6) with roughly halving spatial dims."""
        coeffs = extractor.extract(small_gray)
        for level_idx, c in enumerate(coeffs):
            assert c.ndim == 3, f"Level {level_idx} should be 3-dimensional"
            assert c.shape[2] == 6, f"Level {level_idx} should have 6 orientations"
            # Spatial dimensions should be positive
            assert c.shape[0] > 0 and c.shape[1] > 0

        # Each successive level should have equal or smaller spatial dims
        for i in range(len(coeffs) - 1):
            assert coeffs[i].shape[0] >= coeffs[i + 1].shape[0]
            assert coeffs[i].shape[1] >= coeffs[i + 1].shape[1]

    def test_pe03_coefficients_are_complex(self, extractor, small_gray):
        """PE-03: DT-CWT coefficients must be complex-valued."""
        coeffs = extractor.extract(small_gray)
        for level_idx, c in enumerate(coeffs):
            assert np.iscomplexobj(c), f"Level {level_idx} should be complex"

    def test_pe06_determinism(self, extractor, small_gray):
        """PE-06: Same input produces identical output."""
        coeffs_a = extractor.extract(small_gray)
        coeffs_b = extractor.extract(small_gray)
        for a, b in zip(coeffs_a, coeffs_b):
            np.testing.assert_array_equal(a, b)

    def test_pe07_rgb_input_accepted(self, extractor, small_rgb):
        """PE-07: RGB (H, W, 3) input is accepted and returns valid coefficients."""
        coeffs = extractor.extract(small_rgb)
        assert len(coeffs) == NLEVELS
        for c in coeffs:
            assert c.ndim == 3
            assert c.shape[2] == 6
            assert np.iscomplexobj(c)


class TestPhaseMaps:
    """Tests for PhaseExtractor.get_phase_maps()."""

    def test_pe04_phase_values_in_range(self, extractor, small_gray):
        """PE-04: Phase values lie in [-pi, pi]."""
        phase_maps = extractor.get_phase_maps(small_gray)
        assert len(phase_maps) == NLEVELS
        for level_idx, p in enumerate(phase_maps):
            assert p.min() >= -np.pi - 1e-10, f"Level {level_idx}: values below -pi"
            assert p.max() <= np.pi + 1e-10, f"Level {level_idx}: values above pi"


class TestMagnitudeMaps:
    """Tests for PhaseExtractor.get_magnitude_maps()."""

    def test_pe05_magnitude_non_negative(self, extractor, small_gray):
        """PE-05: Magnitude values are >= 0."""
        mag_maps = extractor.get_magnitude_maps(small_gray)
        assert len(mag_maps) == NLEVELS
        for level_idx, m in enumerate(mag_maps):
            assert m.min() >= -1e-15, f"Level {level_idx}: negative magnitudes"
