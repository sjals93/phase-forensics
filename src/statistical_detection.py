"""Statistical detection module: GGD fitting and Neyman-Pearson test."""

import numpy as np
from scipy.stats import gennorm
from scipy.optimize import minimize_scalar
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GGDFitter:
    """Generalized Gaussian Distribution fitter.

    GGD: p(x; α, β) = (β / 2αΓ(1/β)) * exp(-(|x|/α)^β)
    - α: scale parameter
    - β: shape parameter (β=2: Gaussian, β=1: Laplacian)
    """

    def fit(self, data: np.ndarray) -> Dict[str, float]:
        """Fit GGD to data using MLE.

        Args:
            data: 1D array of values.

        Returns:
            Dict with 'alpha' (scale), 'beta' (shape) parameters.
        """
        try:
            beta, loc, alpha = gennorm.fit(data, floc=0)
            if beta <= 0 or alpha <= 0:
                raise ValueError("Invalid parameters")
        except Exception as e:
            logger.warning(f"GGD fitting failed ({e}), using Laplacian fallback")
            beta = 1.0
            alpha = np.mean(np.abs(data)) if len(data) > 0 else 1.0

        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
        }

    def log_likelihood(
        self, data: np.ndarray, alpha: float, beta: float
    ) -> float:
        """Compute log-likelihood of data under GGD."""
        return np.sum(gennorm.logpdf(data, beta, loc=0, scale=alpha))


class NeymanPearsonDetector:
    """Neyman-Pearson likelihood ratio test for fake detection.

    H0: image is real (GGD parameters from calibration)
    H1: image is fake (GGD parameters deviate from real)
    """

    def __init__(self, real_stats: Dict):
        self.real_stats = real_stats

    def score(self, ispc_value: float, cdpgc_value: float) -> float:
        """Compute detection score.

        Positive score = more likely fake.
        Score is based on Mahalanobis-like distance from real distribution.

        Args:
            ispc_value: Mean ISPC value of test image.
            cdpgc_value: Mean CDPGC value of test image.

        Returns:
            Detection score (higher = more likely fake).
        """
        # Z-score based on real distribution statistics
        ispc_z = (ispc_value - self.real_stats['ispc_mean']) / max(
            self.real_stats['ispc_std'], 1e-8
        )
        cdpgc_z = (cdpgc_value - self.real_stats['cdpgc_mean']) / max(
            self.real_stats['cdpgc_std'], 1e-8
        )

        # Combined score (positive direction = more fake)
        # Fake images have HIGHER ispc and HIGHER cdpgc
        score = (ispc_z + cdpgc_z) / 2.0

        return float(score)

    def score_with_ggd(
        self, data: np.ndarray, metric_name: str
    ) -> float:
        """Score using full GGD likelihood ratio.

        Args:
            data: Feature values from test image.
            metric_name: 'ispc' or 'cdpgc'.

        Returns:
            Log-likelihood ratio.
        """
        ggd_params = self.real_stats[f'{metric_name}_ggd']
        alpha = ggd_params['alpha']
        beta = ggd_params['beta']

        # Log-likelihood under H0 (real)
        ll_real = np.sum(gennorm.logpdf(data, beta, loc=0, scale=alpha))

        # Log-likelihood under H1 (fake) - use test data's own GGD
        try:
            beta_test, _, alpha_test = gennorm.fit(data, floc=0)
            ll_fake = np.sum(
                gennorm.logpdf(data, beta_test, loc=0, scale=alpha_test)
            )
        except Exception:
            ll_fake = ll_real  # Fallback: no discrimination

        # Likelihood ratio (positive = more likely fake)
        return float(ll_fake - ll_real)
