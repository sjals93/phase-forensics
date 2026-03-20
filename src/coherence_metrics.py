"""Phase coherence metrics: ISPC and CDPGC."""

import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# DT-CWT 6 subband orientations in degrees
DTCWT_DIRECTIONS_DEG = np.array([15.0, 45.0, 75.0, 105.0, 135.0, 165.0])
DTCWT_DIRECTIONS_RAD = np.deg2rad(DTCWT_DIRECTIONS_DEG)
# Perpendicular directions (rotate by 90 degrees)
DTCWT_PERP_RAD = DTCWT_DIRECTIONS_RAD + np.pi / 2.0


def _downsample_phase(phase: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Downsample a phase map to target spatial dimensions using circular mean.

    Args:
        phase: Phase array of shape (H, W, 6).
        target_h: Target height.
        target_w: Target width.

    Returns:
        Downsampled phase array of shape (target_h, target_w, 6).
    """
    h, w = phase.shape[:2]
    if h == target_h and w == target_w:
        return phase

    ratio_h = h // target_h
    ratio_w = w // target_w

    if ratio_h > 0 and ratio_w > 0:
        cropped = phase[:target_h * ratio_h, :target_w * ratio_w]
        reshaped = cropped.reshape(target_h, ratio_h, target_w, ratio_w, 6)
        # Circular mean for phase downsampling
        return np.angle(np.mean(np.exp(1j * reshaped), axis=(1, 3)))
    else:
        return phase[:target_h, :target_w]


def compute_ispc(
    phase_maps: List[np.ndarray],
    scale_pairs: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Compute Inter-Scale Phase Coherence (ISPC).

    ISPC(x,y,d) = 1 - |mean_l(exp(j*(phi_{l,d}(x,y) - phi_{l+1,d}(x,y))))|

    The mean is taken over scale pairs l at each spatial location. This measures
    how consistently the phase difference between adjacent scales behaves across
    multiple scale transitions. All phase difference maps are downsampled to the
    coarsest resolution before computing the circular mean.

    Low ISPC = high coherence (real), High ISPC = low coherence (fake).

    Args:
        phase_maps: List of phase arrays per level, each (H_l, W_l, 6).
        scale_pairs: List of (level_i, level_j) pairs. Default: adjacent levels.

    Returns:
        ISPC values per direction, shape (6,). This is the spatial average of
        the ISPC map (which has shape (H_coarse, W_coarse, 6)).

    Raises:
        ValueError: If fewer than 2 levels are provided.
    """
    if len(phase_maps) < 2:
        raise ValueError(
            f"Need at least 2 levels for ISPC, got {len(phase_maps)}"
        )

    if scale_pairs is None:
        scale_pairs = [(i, i + 1) for i in range(len(phase_maps) - 1)]

    # Validate scale pairs
    valid_pairs = [
        (l1, l2) for l1, l2 in scale_pairs
        if l1 < len(phase_maps) and l2 < len(phase_maps)
    ]
    if len(valid_pairs) == 0:
        raise ValueError("No valid scale pairs found.")

    # Determine coarsest resolution among all levels involved in valid_pairs
    levels_involved = set()
    for l1, l2 in valid_pairs:
        levels_involved.add(l1)
        levels_involved.add(l2)

    coarsest_h = min(phase_maps[l].shape[0] for l in levels_involved)
    coarsest_w = min(phase_maps[l].shape[1] for l in levels_involved)

    # Compute phase differences, downsample to coarsest resolution, and stack
    phase_diffs = []  # Each will be (coarsest_h, coarsest_w, 6)

    for l1, l2 in valid_pairs:
        phase1 = phase_maps[l1]
        phase2 = phase_maps[l2]

        # Downsample both to the coarsest resolution
        phase1_ds = _downsample_phase(phase1, coarsest_h, coarsest_w)
        phase2_ds = _downsample_phase(phase2, coarsest_h, coarsest_w)

        # Circular phase difference
        phase_diff = np.angle(np.exp(1j * (phase1_ds - phase2_ds)))
        phase_diffs.append(phase_diff)

    # Stack: shape (n_pairs, H_coarse, W_coarse, 6)
    phase_diff_stack = np.stack(phase_diffs, axis=0)

    # Circular mean over scale pairs at each (x, y, d)
    mean_vector = np.mean(np.exp(1j * phase_diff_stack), axis=0)  # (H, W, 6)

    # ISPC map: circular variance at each pixel per direction
    ispc_map = 1.0 - np.abs(mean_vector)  # (H, W, 6)

    # Return spatial average per direction: shape (6,)
    return np.mean(ispc_map, axis=(0, 1))


def compute_ispc_map(
    phase_maps: List[np.ndarray],
    patch_size: int = 16,
) -> np.ndarray:
    """Compute spatial ISPC map for localization.

    For each patch, accumulates exp(j*phase_diff) across all scale pairs
    (downsampled to coarsest resolution), then computes circular variance
    per direction and averages across directions.

    Args:
        phase_maps: List of phase arrays per level.
        patch_size: Size of local patches for computing local ISPC.

    Returns:
        ISPC map at patch resolution, shape (n_patches_h, n_patches_w).

    Raises:
        ValueError: If fewer than 2 levels are provided.
    """
    if len(phase_maps) < 2:
        raise ValueError("Need at least 2 levels for ISPC")

    scale_pairs = [(i, i + 1) for i in range(len(phase_maps) - 1)]

    # Determine coarsest resolution
    coarsest_h = min(pm.shape[0] for pm in phase_maps)
    coarsest_w = min(pm.shape[1] for pm in phase_maps)

    # Compute and stack phase differences at coarsest resolution
    phase_diffs = []
    for l1, l2 in scale_pairs:
        phase1_ds = _downsample_phase(phase_maps[l1], coarsest_h, coarsest_w)
        phase2_ds = _downsample_phase(phase_maps[l2], coarsest_h, coarsest_w)
        phase_diff = np.angle(np.exp(1j * (phase1_ds - phase2_ds)))
        phase_diffs.append(phase_diff)

    # Stack: (n_pairs, H_coarse, W_coarse, 6)
    phase_diff_stack = np.stack(phase_diffs, axis=0)

    # Circular mean over scale pairs: (H_coarse, W_coarse, 6)
    mean_vector = np.mean(np.exp(1j * phase_diff_stack), axis=0)
    # Per-pixel ISPC per direction
    ispc_full = 1.0 - np.abs(mean_vector)  # (H_coarse, W_coarse, 6)

    # Aggregate into patches
    ps = min(patch_size, coarsest_h, coarsest_w)
    n_patches_h = coarsest_h // ps
    n_patches_w = coarsest_w // ps

    if n_patches_h == 0 or n_patches_w == 0:
        # Image too small for patching; return single-pixel average
        return np.mean(ispc_full, axis=2, keepdims=False)[:1, :1]

    ispc_map = np.zeros((n_patches_h, n_patches_w))

    for i in range(n_patches_h):
        for j in range(n_patches_w):
            patch = ispc_full[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps, :]
            # Mean across spatial pixels and directions within this patch
            ispc_map[i, j] = np.mean(patch)

    return ispc_map


def compute_cdpgc(
    phase_maps: List[np.ndarray],
) -> np.ndarray:
    """Compute Cross-Direction Phase Gradient Consistency (CDPGC).

    CDPGC(x,y,l) = Var_d( nabla_phi_{l,d}(x,y) . e_d_perp )

    For each direction d, the phase gradient is projected onto the perpendicular
    of that wavelet orientation. Then the variance across 6 directions is computed
    at each spatial location, and the result is averaged spatially.

    Low CDPGC = consistent gradients (real), High CDPGC = inconsistent (fake).

    Args:
        phase_maps: List of phase arrays per level, each (H_l, W_l, 6).

    Returns:
        CDPGC values per level, shape (n_levels,).
    """
    cdpgc_values = []

    cos_perp = np.cos(DTCWT_PERP_RAD)  # (6,)
    sin_perp = np.sin(DTCWT_PERP_RAD)  # (6,)

    for level_idx, phase in enumerate(phase_maps):
        h, w, n_dirs = phase.shape

        if h < 3 or w < 3:
            logger.warning(
                f"Level {level_idx}: spatial dims ({h}, {w}) too small for "
                f"gradient computation, returning 0.0"
            )
            cdpgc_values.append(0.0)
            continue

        # Compute projected gradient for each direction at each pixel
        # Common size after finite differences: (h-1, w-1)
        common_h = h - 1
        common_w = w - 1

        # projected_grads: (common_h, common_w, 6)
        projected_grads = np.zeros((common_h, common_w, n_dirs))

        for d in range(n_dirs):
            p = phase[:, :, d]

            # Circular finite difference gradients
            # grad_row: difference along rows (dφ/dy, vertical direction)
            grad_row = np.angle(
                np.exp(1j * p[1:, :]) * np.exp(-1j * p[:-1, :])
            )  # (h-1, w) = dφ/dy
            # grad_col: difference along columns (dφ/dx, horizontal direction)
            grad_col = np.angle(
                np.exp(1j * p[:, 1:]) * np.exp(-1j * p[:, :-1])
            )  # (h, w-1) = dφ/dx

            # Trim to common size
            dphidx = grad_col[:common_h, :common_w]
            dphidy = grad_row[:common_h, :common_w]

            # Project gradient onto perpendicular direction of this subband
            projected_grads[:, :, d] = dphidx * cos_perp[d] + dphidy * sin_perp[d]

        # Variance across directions at each spatial location
        # shape: (common_h, common_w)
        spatial_cdpgc = np.var(projected_grads, axis=2)

        # Average spatially
        cdpgc_values.append(float(np.mean(spatial_cdpgc)))

    return np.array(cdpgc_values)
