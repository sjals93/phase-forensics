"""Tests for PhaseForensicsDetector: integration, contracts, round-trip."""

import tempfile
import os

import numpy as np
import pytest

from src.detector import PhaseForensicsDetector

NLEVELS = 3


@pytest.fixture
def detector():
    return PhaseForensicsDetector(nlevels=NLEVELS)


@pytest.fixture
def small_gray():
    np.random.seed(42)
    return np.random.rand(64, 64).astype(np.float64)


@pytest.fixture
def small_images():
    """A handful of small random images for fitting."""
    rng = np.random.RandomState(42)
    return [rng.rand(64, 64).astype(np.float64) for _ in range(5)]


@pytest.fixture
def fitted_detector(small_images):
    """A detector that has been fit on small_images."""
    det = PhaseForensicsDetector(nlevels=NLEVELS)
    det.fit_real_statistics(small_images)
    return det


class TestExtractFeatures:
    """Tests for PhaseForensicsDetector.extract_features()."""

    def test_pd01_returns_dict_with_required_keys(self, detector, small_gray):
        """PD-01: extract_features returns dict with ispc, cdpgc, ispc_map, ispc_mean, cdpgc_mean."""
        features = detector.extract_features(small_gray)
        assert isinstance(features, dict)
        for key in ('ispc', 'cdpgc', 'ispc_map', 'ispc_mean', 'cdpgc_mean'):
            assert key in features, f"Missing key: {key}"

    def test_pd01_ispc_is_numpy_array(self, detector, small_gray):
        """PD-01b: The ispc value is a numpy array."""
        features = detector.extract_features(small_gray)
        assert isinstance(features['ispc'], np.ndarray)

    def test_pd01_cdpgc_is_numpy_array(self, detector, small_gray):
        """PD-01c: The cdpgc value is a numpy array."""
        features = detector.extract_features(small_gray)
        assert isinstance(features['cdpgc'], np.ndarray)

    def test_pd01_ispc_map_is_2d(self, detector, small_gray):
        """PD-01d: The ispc_map is a 2D array."""
        features = detector.extract_features(small_gray)
        assert features['ispc_map'].ndim == 2

    def test_pd01_scalar_means(self, detector, small_gray):
        """PD-01e: ispc_mean and cdpgc_mean are scalar floats."""
        features = detector.extract_features(small_gray)
        assert np.isscalar(features['ispc_mean']) or features['ispc_mean'].ndim == 0
        assert np.isscalar(features['cdpgc_mean']) or features['cdpgc_mean'].ndim == 0


class TestDetectBeforeFit:
    """Tests for calling detect() before fit_real_statistics()."""

    def test_pd02_detect_before_fit_raises(self, detector, small_gray):
        """PD-02: detect() before fit_real_statistics() raises RuntimeError."""
        with pytest.raises(RuntimeError):
            detector.detect(small_gray)


class TestDetectEndToEnd:
    """Tests for fit + detect end-to-end pipeline."""

    def test_pd03_fit_detect_works(self, fitted_detector, small_gray):
        """PD-03: fit_real_statistics + detect completes without error."""
        result = fitted_detector.detect(small_gray)
        assert isinstance(result, dict)
        assert 'score' in result
        assert 'is_fake' in result
        assert 'features' in result
        assert 'ispc_map' in result

    def test_pd04_is_fake_consistent_with_score(self, fitted_detector, small_gray):
        """PD-04: is_fake == True iff score > 0."""
        result = fitted_detector.detect(small_gray)
        if result['score'] > 0:
            assert result['is_fake'] is True or result['is_fake'] == True
        elif result['score'] <= 0:
            assert result['is_fake'] is False or result['is_fake'] == False

    def test_pd03_score_is_float(self, fitted_detector, small_gray):
        """PD-03b: The score is a numeric float."""
        result = fitted_detector.detect(small_gray)
        assert isinstance(result['score'], (float, np.floating))


class TestSaveLoadRoundTrip:
    """Tests for save_model / load_model round-trip."""

    def test_pd05_round_trip_same_scores(self, fitted_detector, small_gray):
        """PD-05: save + load produces a detector that gives the same score."""
        result_before = fitted_detector.detect(small_gray)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.npy")
            fitted_detector.save_model(model_path)

            loaded_detector = PhaseForensicsDetector(nlevels=NLEVELS)
            loaded_detector.load_model(model_path)

            result_after = loaded_detector.detect(small_gray)

        np.testing.assert_allclose(
            result_before['score'],
            result_after['score'],
            rtol=1e-12,
            err_msg="Score should be identical after save/load round-trip",
        )

    def test_pd05_save_before_fit_raises(self, detector):
        """PD-05b: save_model before fit raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.npy")
            with pytest.raises(RuntimeError):
                detector.save_model(model_path)
