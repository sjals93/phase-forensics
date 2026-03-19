"""Pilot Study: GO/NO-GO decision for PhaseForensics.

This script is the FIRST thing to run. It determines whether the phase
coherence signal is strong enough to proceed with the full project.

GO criteria:
- Cohen's d > 0.5 for at least 7/9 generators → GO
- Cohen's d = 0.3~0.5 → CONDITIONAL GO (amplitude+phase combination)
- Cohen's d < 0.3 → NO-GO (switch to FoMFP or FAP)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
import json
from datetime import datetime

from src.phase_extraction import PhaseExtractor
from src.coherence_metrics import compute_ispc, compute_cdpgc

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return abs(np.mean(group1) - np.mean(group2)) / max(pooled_std, 1e-8)


def load_images_from_dir(
    directory: str, n_samples: int = 100, extensions: tuple = ('.png', '.jpg', '.jpeg')
) -> List[np.ndarray]:
    """Load images from directory."""
    from PIL import Image

    path = Path(directory)
    if not path.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    files = [f for f in path.iterdir() if f.suffix.lower() in extensions]
    files = sorted(files)[:n_samples]

    images = []
    for f in files:
        try:
            img = np.array(Image.open(f).convert('RGB')) / 255.0
            images.append(img)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    return images


def run_pilot_study(
    data_config: Dict[str, str],
    n_samples: int = 100,
    output_dir: str = 'results/pilot_study',
):
    """Run the pilot study.

    Args:
        data_config: Dict mapping generator names to image directories.
        n_samples: Number of images per generator.
        output_dir: Where to save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    extractor = PhaseExtractor(nlevels=5)

    results = {}

    for name, directory in data_config.items():
        logger.info(f"Processing {name}...")
        images = load_images_from_dir(directory, n_samples)

        if not images:
            logger.warning(f"  No images found for {name}, skipping")
            continue

        ispc_values = []
        cdpgc_values = []

        for i, img in enumerate(images):
            phase_maps = extractor.get_phase_maps(img)
            ispc = compute_ispc(phase_maps)
            cdpgc = compute_cdpgc(phase_maps)
            ispc_values.append(np.mean(ispc))
            cdpgc_values.append(np.mean(cdpgc))

        results[name] = {
            'ispc_values': ispc_values,
            'cdpgc_values': cdpgc_values,
            'ispc_mean': float(np.mean(ispc_values)),
            'ispc_std': float(np.std(ispc_values)),
            'cdpgc_mean': float(np.mean(cdpgc_values)),
            'cdpgc_std': float(np.std(cdpgc_values)),
            'n_images': len(images),
        }
        logger.info(
            f"  {name}: ISPC={results[name]['ispc_mean']:.4f}±{results[name]['ispc_std']:.4f}, "
            f"CDPGC={results[name]['cdpgc_mean']:.6f}±{results[name]['cdpgc_std']:.6f}"
        )

    # Compute Cohen's d for each generator vs real
    if 'real' not in results:
        logger.error("No 'real' images found. Cannot compute Cohen's d.")
        return

    real_ispc = np.array(results['real']['ispc_values'])
    real_cdpgc = np.array(results['real']['cdpgc_values'])

    logger.info("\n" + "=" * 60)
    logger.info("GO/NO-GO DECISION")
    logger.info("=" * 60)

    go_count = 0
    total_generators = 0
    decision_table = []

    for name, data in results.items():
        if name == 'real':
            continue
        total_generators += 1

        fake_ispc = np.array(data['ispc_values'])
        fake_cdpgc = np.array(data['cdpgc_values'])

        d_ispc = cohens_d(real_ispc, fake_ispc)
        d_cdpgc = cohens_d(real_cdpgc, fake_cdpgc)
        d_combined = max(d_ispc, d_cdpgc)

        if d_combined > 0.5:
            status = "GO"
            go_count += 1
        elif d_combined > 0.3:
            status = "CONDITIONAL"
        else:
            status = "WEAK"

        decision_table.append({
            'generator': name,
            'd_ispc': d_ispc,
            'd_cdpgc': d_cdpgc,
            'd_combined': d_combined,
            'status': status,
        })
        logger.info(
            f"  {name:20s}: d_ispc={d_ispc:.3f}, d_cdpgc={d_cdpgc:.3f}, "
            f"d_max={d_combined:.3f} [{status}]"
        )

    # Final decision
    logger.info("\n" + "-" * 60)
    go_ratio = go_count / max(total_generators, 1)

    if go_ratio >= 7 / 9:
        final = "GO"
        logger.info(f"FINAL DECISION: *** GO *** ({go_count}/{total_generators} generators pass)")
    elif go_ratio >= 5 / 9:
        final = "CONDITIONAL GO"
        logger.info(f"FINAL DECISION: CONDITIONAL GO ({go_count}/{total_generators})")
        logger.info("  → Consider amplitude+phase combination")
    else:
        final = "NO-GO"
        logger.info(f"FINAL DECISION: NO-GO ({go_count}/{total_generators})")
        logger.info("  → Switch to FoMFP or FAP")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'n_samples_per_generator': n_samples,
        'decision': final,
        'go_count': go_count,
        'total_generators': total_generators,
        'decision_table': decision_table,
        'stats': {k: {kk: vv for kk, vv in v.items() if kk != 'ispc_values' and kk != 'cdpgc_values'}
                  for k, v in results.items()},
    }

    output_path = os.path.join(output_dir, 'pilot_study_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    # TODO: Update paths to actual dataset locations
    data_config = {
        'real': '/mnt/storage/datasets/GeneralDatasets/COCO/val2017',
        # GAN
        # 'progan': '/mnt/storage/datasets/DeepfakeDetection/Image/CNNDetection/progan',
        # 'stylegan2': '/mnt/storage/datasets/DeepfakeDetection/Image/GenImage/stylegan2',
        # Diffusion
        # 'sd15': '/mnt/storage/datasets/DeepfakeDetection/Image/GenImage/stable_diffusion_v_1_5',
        # 'sdxl': '/mnt/storage/datasets/DeepfakeDetection/Image/GenImage/sdxl',
        # 'midjourney': '/mnt/storage/datasets/DeepfakeDetection/Image/GenImage/midjourney',
    }

    # Filter to existing directories only
    existing = {k: v for k, v in data_config.items() if Path(v).exists()}

    if len(existing) < 2:
        logger.error("Need at least 'real' + 1 generator directory. Update data_config paths.")
        logger.info("Available datasets:")
        for p in Path('/mnt/storage/datasets/').rglob('*'):
            if p.is_dir() and not any(x.startswith('.') for x in p.parts):
                logger.info(f"  {p}")
    else:
        run_pilot_study(existing, n_samples=100)
