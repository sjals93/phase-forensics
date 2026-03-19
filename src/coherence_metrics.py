"""Phase coherence metrics: ISPC and CDPGC."""

import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_ispc(
    phase_maps: List[np.ndarray],
    scale_pairs: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Compute Inter-Scale Phase Coherence (ISPC).

    Measures circular variance of phase differences between adjacent scales.
    Low ISPC = high coherence (real), High ISPC = low coherence (fake).

    Args:
        phase_maps: List of phase arrays per level, each (H_l, W_l, 6).
        scale_pairs: List of (level_i, level_j) pairs. Default: adjacent levels.

    Returns:
        ISPC values, shape depends on aggregation.
    """
    if scale_pairs is None:
        scale_pairs = [(i, i + 1) for i in range(len(phase_maps) - 1)]

    ispc_values = []

    for l1, l2 in scale_pairs:
        if l1 >= len(phase_maps) or l2 >= len(phase_maps):
            continue

        phase1 = phase_maps[l1]  # (H1, W1, 6)
        phase2 = phase_maps[l2]  # (H2, W2, 6)

        # Resize to match smaller scale
        h2, w2 = phase2.shape[:2]
        # Downsample phase1 to match phase2 dimensions
        h1, w1 = phase1.shape[:2]
        if h1 != h2 or w1 != w2:
            # Simple block averaging for downsampling
            ratio_h = h1 // h2
            ratio_w = w1 // w2
            if ratio_h > 0 and ratio_w > 0:
                phase1_ds = phase1[:h2 * ratio_h, :w2 * ratio_w].reshape(
                    h2, ratio_h, w2, ratio_w, 6
                )
                # Circular mean for phase downsampling
                phase1_ds = np.angle(
                    np.mean(np.exp(1j * phase1_ds), axis=(1, 3))
                )
            else:
                phase1_ds = phase1[:h2, :w2]
        else:
            phase1_ds = phase1

        # Phase difference (circular)
        phase_diff = np.angle(np.exp(1j * (phase1_ds - phase2)))

        # Circular variance per direction, then average across directions
        # CV = 1 - |mean(exp(j * phase_diff))|
        mean_vector = np.mean(np.exp(1j * phase_diff), axis=(0, 1))  # (6,)
        cv_per_dir = 1.0 - np.abs(mean_vector)
        ispc_values.append(cv_per_dir)

    return np.array(ispc_values)  # (n_pairs, 6)


def compute_ispc_map(
    phase_maps: List[np.ndarray],
    patch_size: int = 16,
) -> np.ndarray:
    """Compute spatial ISPC map for localization.

    Args:
        phase_maps: List of phase arrays per level.
        patch_size: Size of local patches for computing local ISPC.

    Returns:
        ISPC map at the resolution of the second-finest level.
    """
    if len(phase_maps) < 2:
        raise ValueError("Need at least 2 levels for ISPC")

    phase1 = phase_maps[0]  # Finest level
    phase2 = phase_maps[1]

    h2, w2 = phase2.shape[:2]
    h1, w1 = phase1.shape[:2]

    # Downsample phase1
    ratio_h = max(1, h1 // h2)
    ratio_w = max(1, w1 // w2)

    if ratio_h > 1 and ratio_w > 1:
        phase1_ds = phase1[:h2 * ratio_h, :w2 * ratio_w].reshape(
            h2, ratio_h, w2, ratio_w, 6
        )
        phase1_ds = np.angle(np.mean(np.exp(1j * phase1_ds), axis=(1, 3)))
    else:
        phase1_ds = phase1[:h2, :w2]

    # Local circular variance in patches
    phase_diff = np.angle(np.exp(1j * (phase1_ds - phase2)))

    ps = min(patch_size, h2, w2)
    n_patches_h = h2 // ps
    n_patches_w = w2 // ps

    ispc_map = np.zeros((n_patches_h, n_patches_w))

    for i in range(n_patches_h):
        for j in range(n_patches_w):
            patch = phase_diff[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps, :]
            mean_vec = np.mean(np.exp(1j * patch))
            ispc_map[i, j] = 1.0 - np.abs(mean_vec)

    return ispc_map


def compute_cdpgc(
    phase_maps: List[np.ndarray],
    gradient_method: str = 'sobel',
) -> np.ndarray:
    """Compute Cross-Direction Phase Gradient Consistency (CDPGC).

    Measures variance of phase gradients across different wavelet directions.
    Low CDPGC = consistent gradients (real), High CDPGC = inconsistent (fake).

    Args:
        phase_maps: List of phase arrays per level, each (H_l, W_l, 6).
        gradient_method: 'sobel' or 'finite_diff'.

    Returns:
        CDPGC values per level, shape (n_levels, ).
    """
    cdpgc_values = []

    for level_idx, phase in enumerate(phase_maps):
        h, w, n_dirs = phase.shape

        if h < 3 or w < 3:
            cdpgc_values.append(0.0)
            continue

        # Compute phase gradient magnitude per direction
        grad_magnitudes = []
        for d in range(n_dirs):
            p = phase[:, :, d]
            # Circular gradient using complex exponential
            # grad = angle(exp(j*p[x+1]) * conj(exp(j*p[x])))
            grad_x = np.angle(np.exp(1j * p[1:, :]) * np.exp(-1j * p[:-1, :]))
            grad_y = np.angle(np.exp(1j * p[:, 1:]) * np.exp(-1j * p[:, :-1]))

            # Gradient magnitude (trimmed to common size)
            min_h = min(grad_x.shape[0], grad_y.shape[0])
            min_w = min(grad_x.shape[1], grad_y.shape[1])
            grad_mag = np.sqrt(
                grad_x[:min_h, :min_w] ** 2 + grad_y[:min_h, :min_w] ** 2
            )
            grad_magnitudes.append(np.mean(grad_mag))

        # CDPGC = variance of gradient magnitudes across directions
        cdpgc_values.append(np.var(grad_magnitudes))

    return np.array(cdpgc_values)
