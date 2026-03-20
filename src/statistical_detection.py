"""Statistical detection module: GGD fitting and Neyman-Pearson test."""

import numpy as np
from scipy.stats import gennorm
from scipy.optimize import minimize_scalar
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GGDFitter:
    """Generalized Gaussian Distribution fitter.

    GGD: p(x; alpha, beta) = (beta / 2*alpha*Gamma(1/beta)) * exp(-(|x|/alpha)^beta)
    - alpha: scale parameter
    - beta: shape parameter (beta=2: Gaussian, beta=1: Laplacian)

    Data is centered before fitting (mean is subtracted and stored separately),
    so the symmetric GGD assumption holds even for non-negative inputs like
    ISPC values in [0, 1] or CDPGC values >= 0.
    """

    def fit(self, data: np.ndarray) -> Dict[str, float]:
        """Fit GGD to data using MLE with proper centering.

        The data is centered (mean-subtracted) before fitting the symmetric GGD
        with floc=0. The mean is stored so it can be used when computing
        log-likelihoods later.

        Args:
            data: 1D array of values.

        Returns:
            Dict with 'alpha' (scale), 'beta' (shape), 'center' (data mean),
            'mean', 'std' parameters.
        """
        data = np.asarray(data).ravel()
        center = float(np.mean(data))
        centered_data = data - center

        try:
            beta, loc, alpha = gennorm.fit(centered_data, floc=0)
            if beta <= 0 or alpha <= 0:
                raise ValueError(f"Invalid GGD parameters: beta={beta}, alpha={alpha}")
        except Exception as e:
            logger.warning(f"GGD fitting failed ({e}), using Laplacian fallback")
            beta = 1.0
            alpha = float(np.mean(np.abs(centered_data))) if len(centered_data) > 0 else 1.0

        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'center': center,
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
        }

    def log_likelihood(
        self, data: np.ndarray, alpha: float, beta: float, center: float = 0.0
    ) -> float:
        """Compute log-likelihood of data under GGD with centering.

        Args:
            data: 1D array of observed values.
            alpha: GGD scale parameter.
            beta: GGD shape parameter.
            center: The mean used for centering during fitting.

        Returns:
            Total log-likelihood.
        """
        centered = np.asarray(data).ravel() - center
        return float(np.sum(gennorm.logpdf(centered, beta, loc=0, scale=alpha)))


class NeymanPearsonDetector:
    """Neyman-Pearson likelihood ratio test for fake detection.

    H0: image is real (GGD parameters from calibration)
    H1: image is fake (GGD parameters deviate from real)

    Primary scoring uses GGD log-likelihood ratio. Z-score scoring is
    available as a fallback.
    """

    def __init__(self, real_stats: Dict):
        self.real_stats = real_stats
        self._ggd_fitter = GGDFitter()

    def score(self, ispc_value: float, cdpgc_value: float) -> float:
        """Compute detection score using z-score (fallback method).

        Positive score = more likely fake.

        Args:
            ispc_value: Mean ISPC value of test image.
            cdpgc_value: Mean CDPGC value of test image.

        Returns:
            Detection score (higher = more likely fake).
        """
        ispc_z = (ispc_value - self.real_stats['ispc_mean']) / max(
            self.real_stats['ispc_std'], 1e-8
        )
        cdpgc_z = (cdpgc_value - self.real_stats['cdpgc_mean']) / max(
            self.real_stats['cdpgc_std'], 1e-8
        )

        # Combined score (positive direction = more fake)
        return float((ispc_z + cdpgc_z) / 2.0)

    def score_with_ggd(
        self, data: np.ndarray, metric_name: str
    ) -> float:
        """Score using full GGD likelihood ratio.

        Computes the log-likelihood ratio: LL(H1_fake) - LL(H0_real).
        If the test data fits its own GGD better than the real GGD,
        the score is positive (more likely fake).

        Args:
            data: Feature values from test image (1D array).
            metric_name: 'ispc' or 'cdpgc'.

        Returns:
            Log-likelihood ratio (positive = more likely fake).
        """
        ggd_params = self.real_stats[f'{metric_name}_ggd']
        alpha = ggd_params['alpha']
        beta = ggd_params['beta']
        center = ggd_params.get('center', 0.0)

        centered_data = np.asarray(data).ravel() - center

        # Log-likelihood under H0 (real distribution)
        ll_real = float(np.sum(gennorm.logpdf(centered_data, beta, loc=0, scale=alpha)))

        # Log-likelihood under H1 (test data's own GGD)
        try:
            beta_test, _, alpha_test = gennorm.fit(centered_data, floc=0)
            if beta_test <= 0 or alpha_test <= 0:
                raise ValueError("Invalid test GGD parameters")
            ll_fake = float(np.sum(
                gennorm.logpdf(centered_data, beta_test, loc=0, scale=alpha_test)
            ))
        except Exception:
            ll_fake = ll_real  # Fallback: no discrimination

        # Likelihood ratio (positive = more likely fake), clamped to avoid inf
        return float(np.clip(ll_fake - ll_real, -1e6, 1e6))

    def score_ggd_combined(
        self,
        ispc_features: np.ndarray,
        cdpgc_features: np.ndarray,
    ) -> float:
        """Compute combined GGD-based Neyman-Pearson score.

        This is the primary scoring method. It computes the GGD log-likelihood
        ratio for both ISPC and CDPGC features and returns their average.

        Args:
            ispc_features: ISPC feature values (1D array, e.g. per-direction values).
            cdpgc_features: CDPGC feature values (1D array, e.g. per-level values).

        Returns:
            Combined detection score (higher = more likely fake).
        """
        ispc_llr = self.score_with_ggd(ispc_features, 'ispc')
        cdpgc_llr = self.score_with_ggd(cdpgc_features, 'cdpgc')

        return float((ispc_llr + cdpgc_llr) / 2.0)
