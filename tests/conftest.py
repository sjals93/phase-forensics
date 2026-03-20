"""Shared fixtures for PhaseForensics tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Compatibility shims: dtcwt uses functions removed in NumPy 2.0
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
if not hasattr(np, "issubsctype"):
    np.issubsctype = np.issubdtype

import pytest


@pytest.fixture
def random_seed():
    np.random.seed(42)
    return 42


@pytest.fixture
def random_image_256(random_seed):
    return np.random.rand(256, 256).astype(np.float64)


@pytest.fixture
def random_rgb_256(random_seed):
    return np.random.rand(256, 256, 3).astype(np.float64)


@pytest.fixture
def constant_image():
    return np.full((256, 256), 0.5, dtype=np.float64)


@pytest.fixture
def zero_image():
    return np.zeros((256, 256), dtype=np.float64)


@pytest.fixture
def gradient_image():
    """Smooth horizontal gradient - should have high phase coherence."""
    grad = np.linspace(0, 1, 256, dtype=np.float64)
    return np.tile(grad, (256, 1))


@pytest.fixture
def noise_image(random_seed):
    """Uniform noise - should have low phase coherence."""
    return np.random.rand(256, 256).astype(np.float64)
