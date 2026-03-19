"""DT-CWT based phase extraction module."""

import numpy as np
import dtcwt
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class PhaseExtractor:
    """Extract multi-scale phase information using DT-CWT."""

    def __init__(
        self,
        nlevels: int = 5,
        biort: str = 'near_sym_b',
        qshift: str = 'qshift_b',
    ):
        self.nlevels = nlevels
        self.transform = dtcwt.Transform2d(biort=biort, qshift=qshift)

    def extract(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract phase maps from image.

        Args:
            image: Grayscale image (H, W), float64, range [0, 1].

        Returns:
            List of complex coefficient arrays, one per level.
            Each array has shape (H_l, W_l, 6) for 6 directions.
        """
        if image.ndim == 3:
            # Convert to grayscale
            image = np.mean(image, axis=2)

        result = self.transform.forward(image, nlevels=self.nlevels)
        return result.highpasses

    def get_phase_maps(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract phase-only maps.

        Returns:
            List of phase arrays per level, each (H_l, W_l, 6).
        """
        coeffs = self.extract(image)
        return [np.angle(c) for c in coeffs]

    def get_magnitude_maps(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract magnitude maps (for amplitude baseline comparison).

        Returns:
            List of magnitude arrays per level, each (H_l, W_l, 6).
        """
        coeffs = self.extract(image)
        return [np.abs(c) for c in coeffs]
