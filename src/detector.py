"""PhaseForensics detector: combines phase extraction, metrics, and statistical detection."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Iterable, Optional, Union
import logging
from PIL import Image

from .phase_extraction import PhaseExtractor
from .coherence_metrics import compute_ispc, compute_cdpgc, compute_ispc_map
from .statistical_detection import GGDFitter, NeymanPearsonDetector
from .config import load_config

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

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "PhaseForensicsDetector":
        """Create a detector from a YAML configuration file.

        Args:
            config_path: Path to a YAML config (e.g. configs/default.yaml).

        Returns:
            Configured PhaseForensicsDetector instance.
        """
        cfg = load_config(config_path)
        dtcwt_cfg = cfg.get("dtcwt", {})
        return cls(
            nlevels=dtcwt_cfg.get("nlevels", 5),
            biort=dtcwt_cfg.get("biort", "near_sym_b"),
            qshift=dtcwt_cfg.get("qshift", "qshift_b"),
        )

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

    def fit_real_statistics(self, real_images: Iterable[np.ndarray]) -> None:
        """Fit GGD parameters on real images (one-time calibration).

        Collects ISPC (per-direction, 6 values) and CDPGC (per-level) features
        from each real image, then fits centered GGD distributions on the
        pooled feature values.

        Args:
            real_images: Iterable of real image arrays.  Images are processed
                one at a time so that the full collection need not reside in
                memory simultaneously.
        """
        ispc_all: list[np.ndarray] = []
        cdpgc_all: list[np.ndarray] = []
        ispc_means: list[float] = []
        cdpgc_means: list[float] = []

        for i, img in enumerate(real_images):
            features = self.extract_features(img)
            # Collect raw per-direction/per-level values for GGD fitting
            ispc_all.append(features['ispc'])       # (6,)
            cdpgc_all.append(features['cdpgc'])     # (n_levels,)
            ispc_means.append(float(features['ispc_mean']))
            cdpgc_means.append(float(features['cdpgc_mean']))

            if (i + 1) % 100 == 0:
                logger.info("  Processed %d images", i + 1)

        n_images = len(ispc_means)
        if n_images == 0:
            raise ValueError("real_images iterable was empty; cannot fit statistics.")

        logger.info("Fitting real statistics on %d images...", n_images)

        # Flatten all per-direction ISPC values for GGD fitting
        ispc_flat = np.concatenate(ispc_all)
        cdpgc_flat = np.concatenate(cdpgc_all)

        ispc_mean_arr = np.array(ispc_means)
        cdpgc_mean_arr = np.array(cdpgc_means)

        self._real_stats = {
            'ispc_ggd': self.ggd_fitter.fit(ispc_flat),
            'cdpgc_ggd': self.ggd_fitter.fit(cdpgc_flat),
            'ispc_mean': float(np.mean(ispc_mean_arr)),
            'ispc_std': float(np.std(ispc_mean_arr)),
            'cdpgc_mean': float(np.mean(cdpgc_mean_arr)),
            'cdpgc_std': float(np.std(cdpgc_mean_arr)),
        }

        self.np_detector = NeymanPearsonDetector(self._real_stats)
        logger.info("Real statistics fitted successfully.")

    def detect(self, image: np.ndarray, method: str = 'ggd') -> Dict:
        """Detect if an image is AI-generated.

        Args:
            image: RGB or grayscale image, float64 [0, 1].
            method: Scoring method. 'ggd' for GGD likelihood ratio (default),
                    'zscore' for z-score fallback.

        Returns:
            Dict with 'score', 'is_fake', 'features', 'ispc_map', 'method'.
        """
        if self._real_stats is None:
            raise RuntimeError("Call fit_real_statistics() first.")

        features = self.extract_features(image)

        if method == 'ggd':
            # Primary method: GGD Neyman-Pearson likelihood ratio
            score = self.np_detector.score_ggd_combined(
                ispc_features=features['ispc'],
                cdpgc_features=features['cdpgc'],
            )
        elif method == 'zscore':
            # Fallback: z-score based scoring
            score = self.np_detector.score(
                features['ispc_mean'], features['cdpgc_mean']
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'ggd' or 'zscore'.")

        return {
            'score': score,
            'is_fake': score > 0,  # Simple threshold at 0
            'features': features,
            'ispc_map': features['ispc_map'],
            'method': method,
        }

    def detect_from_path(self, image_path: str, method: str = 'ggd') -> Dict:
        """Detect from file path.

        Args:
            image_path: Path to the image file.
            method: Scoring method ('ggd' or 'zscore').

        Returns:
            Detection result dict.
        """
        img = np.array(Image.open(image_path).convert('RGB')) / 255.0
        return self.detect(img, method=method)

    def save_model(self, path: Union[str, Path]) -> None:
        """Save fitted statistics to a JSON file.

        Args:
            path: Destination file path (should end with .json).

        Raises:
            RuntimeError: If the detector has not been fitted yet.
        """
        if self._real_stats is None:
            raise RuntimeError(
                "Detector has not been fitted. Call fit_real_statistics() before saving."
            )

        path = Path(path)
        with open(path, "w") as f:
            json.dump(self._real_stats, f, indent=2)
        logger.info("Model saved to %s", path)

    def load_model(self, path: Union[str, Path]) -> None:
        """Load pre-fitted statistics from a JSON file.

        Args:
            path: Path to a JSON file previously created by save_model().
        """
        path = Path(path)
        with open(path, "r") as f:
            self._real_stats = json.load(f)
        self.np_detector = NeymanPearsonDetector(self._real_stats)
        logger.info("Model loaded from %s", path)
