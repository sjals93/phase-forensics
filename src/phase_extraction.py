"""DT-CWT based phase extraction module."""

import numpy as np
import dtcwt
from typing import List
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

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        """Validate and normalise an input image.

        Checks:
            - ndim is 2 (H, W) or 3 (H, W, 3)
            - Minimum spatial size >= 2**nlevels per dimension
            - dtype is float with values in [0, 1]; converts if needed
            - Warns on all-zero images

        Returns:
            Validated (and possibly converted) image array.
        """
        # --- shape validation ---
        if image.ndim == 3:
            if image.shape[2] != 3:
                raise ValueError(
                    f"3D image must have 3 channels (H, W, 3), "
                    f"got shape {image.shape}"
                )
        elif image.ndim != 2:
            raise ValueError(
                f"Image must be 2D (H, W) or 3D (H, W, 3), "
                f"got ndim={image.ndim}"
            )

        # --- minimum size validation ---
        min_dim = 2 ** self.nlevels
        h, w = image.shape[:2]
        if h < min_dim or w < min_dim:
            raise ValueError(
                f"Image spatial dimensions ({h}, {w}) are too small for "
                f"nlevels={self.nlevels}. Minimum dimension is {min_dim}."
            )

        # --- dtype / range validation ---
        if not np.issubdtype(image.dtype, np.floating):
            logger.warning(
                "Image dtype is %s, converting to float64 and scaling to [0, 1].",
                image.dtype,
            )
            image = image.astype(np.float64) / 255.0
        else:
            vmin, vmax = float(image.min()), float(image.max())
            if vmin < 0.0 or vmax > 1.0:
                logger.warning(
                    "Image values outside [0, 1] (min=%.4f, max=%.4f). "
                    "Clipping to [0, 1].",
                    vmin,
                    vmax,
                )
                image = np.clip(image, 0.0, 1.0)

        # --- all-zero check ---
        if not image.any():
            logger.warning("Input image is all zeros.")

        return image

    def extract(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract phase maps from image.

        Args:
            image: Grayscale image (H, W) or RGB (H, W, 3),
                   float64, range [0, 1].

        Returns:
            List of complex coefficient arrays, one per level.
            Each array has shape (H_l, W_l, 6) for 6 directions.
        """
        image = self._validate_image(image)

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
