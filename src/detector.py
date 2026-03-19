"""PhaseForensics detector: combines phase extraction, metrics, and statistical detection."""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from PIL import Image

from .phase_extraction import PhaseExtractor
from .coherence_metrics import compute_ispc, compute_cdpgc, compute_ispc_map
from .statistical_detection import GGDFitter, NeymanPearsonDetector

logger = logging.getLogger(__name__)


class PhaseForensicsDetector:
    """Main detector class combining all components."""

    def __init__(
        self,
        nlevels: int = 5,
        biort: str = 'near_sym_b',
        qshift: str = 'qshift_b',
    ):
        self.extractor = PhaseExtractor(
            nlevels=nlevels, biort=biort, qshift=qshift
        )
        self.ggd_fitter = GGDFitter()
        self.np_detector: Optional[NeymanPearsonDetector] = None
        self._real_stats: Optional[Dict] = None

    def extract_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract phase coherence features from a single image.

        Args:
            image: RGB or grayscale image, float64 [0, 1].

        Returns:
            Dict with 'ispc', 'cdpgc', 'ispc_map' keys.
        """
        phase_maps = self.extractor.get_phase_maps(image)

        ispc = compute_ispc(phase_maps)
        cdpgc = compute_cdpgc(phase_maps)
        ispc_map = compute_ispc_map(phase_maps)

        return {
            'ispc': ispc,
            'cdpgc': cdpgc,
            'ispc_map': ispc_map,
            'ispc_mean': np.mean(ispc),
            'cdpgc_mean': np.mean(cdpgc),
        }

    def fit_real_statistics(self, real_images: list) -> None:
        """Fit GGD parameters on real images (one-time calibration).

        Args:
            real_images: List of real image arrays.
        """
        logger.info(f"Fitting real statistics on {len(real_images)} images...")

        ispc_all = []
        cdpgc_all = []

        for i, img in enumerate(real_images):
            features = self.extract_features(img)
            ispc_all.append(features['ispc_mean'])
            cdpgc_all.append(features['cdpgc_mean'])

            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(real_images)}")

        ispc_arr = np.array(ispc_all)
        cdpgc_arr = np.array(cdpgc_all)

        self._real_stats = {
            'ispc_ggd': self.ggd_fitter.fit(ispc_arr),
            'cdpgc_ggd': self.ggd_fitter.fit(cdpgc_arr),
            'ispc_mean': np.mean(ispc_arr),
            'ispc_std': np.std(ispc_arr),
            'cdpgc_mean': np.mean(cdpgc_arr),
            'cdpgc_std': np.std(cdpgc_arr),
        }

        self.np_detector = NeymanPearsonDetector(self._real_stats)
        logger.info("Real statistics fitted successfully.")

    def detect(self, image: np.ndarray) -> Dict:
        """Detect if an image is AI-generated.

        Args:
            image: RGB or grayscale image, float64 [0, 1].

        Returns:
            Dict with 'score', 'is_fake', 'features', 'ispc_map'.
        """
        if self._real_stats is None:
            raise RuntimeError("Call fit_real_statistics() first.")

        features = self.extract_features(image)

        score = self.np_detector.score(
            features['ispc_mean'], features['cdpgc_mean']
        )

        return {
            'score': score,
            'is_fake': score > 0,  # Simple threshold at 0
            'features': features,
            'ispc_map': features['ispc_map'],
        }

    def detect_from_path(self, image_path: str) -> Dict:
        """Detect from file path."""
        img = np.array(Image.open(image_path).convert('RGB')) / 255.0
        return self.detect(img)

    def save_model(self, path: str) -> None:
        """Save fitted statistics."""
        if self._real_stats is None:
            raise RuntimeError("No fitted statistics to save.")
        np.save(path, self._real_stats)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load pre-fitted statistics."""
        self._real_stats = np.load(path, allow_pickle=True).item()
        self.np_detector = NeymanPearsonDetector(self._real_stats)
        logger.info(f"Model loaded from {path}")
