"""Pilot Study: GO/NO-GO decision for PhaseForensics.

This script is the FIRST thing to run. It determines whether the phase
coherence signal is strong enough to proceed with the full project.

GO criteria (7 generators):
- Cohen's d > 0.5 for at least 5/7 generators -> GO
- Cohen's d > 0.5 for 3-4/7 generators -> CONDITIONAL GO (amplitude+phase combination)
- Cohen's d > 0.5 for <3/7 generators -> NO-GO (switch to FoMFP or FAP)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import io
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import yaml

from src.phase_extraction import PhaseExtractor
from src.coherence_metrics import compute_ispc, compute_cdpgc

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAKE_BASE = '/mnt/storage/datasets/DeepfakeDetection/AI_Generated'
REAL_DIR = '/mnt/storage/datasets/DeepfakeDetection/Image/FFHQ/images'
TARGET_SIZE: Tuple[int, int] = (256, 256)

GENERATOR_MAP: Dict[str, str] = {
    'SD_v1.1': os.path.join(FAKE_BASE, 'CompVis-stable-diffusion-v1-1-ViT-L-14-openai'),
    'SD_v1.5': os.path.join(FAKE_BASE, 'runwayml-stable-diffusion-v1-5-ViT-L-14-openai'),
    'SD_v2.1': os.path.join(FAKE_BASE, 'stabilityai-stable-diffusion-2-1-base-ViT-H-14-laion2b_s32b_b79k'),
    'Kandinsky_2.1': os.path.join(FAKE_BASE, 'kandinsky-community-kandinsky-2-1-ViT-L-14-openai'),
    'Midjourney_v4': os.path.join(FAKE_BASE, 'midjourney-v4'),
    'Midjourney_v5': os.path.join(FAKE_BASE, 'midjourney-v5'),
    'Midjourney_v5.1': os.path.join(FAKE_BASE, 'midjourney-v5-1'),
}

# GO/NO-GO thresholds for 7 generators
GO_THRESHOLD = 5          # >= 5/7 with d > 0.5
CONDITIONAL_THRESHOLD = 3  # 3-4/7
TOTAL_GENERATORS = 7
COHENS_D_CUTOFF = 0.5

# Email settings
EMAIL_FROM = 'sjals93@gmail.com'
EMAIL_APP_PASSWORD = 'ibja bxch bcmr cith'
EMAIL_SMTP_HOST = 'smtp.gmail.com'
EMAIL_SMTP_PORT = 587


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return abs(np.mean(group1) - np.mean(group2)) / max(pooled_std, 1e-8)


def resize_image(img: Image.Image, size: Tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """Resize an image to *size* using LANCZOS resampling."""
    if img.size != (size[0], size[1]):
        img = img.resize(size, Image.LANCZOS)
    return img


def load_images_from_dir(
    directory: str,
    n_samples: int = 100,
    extensions: tuple = ('.png', '.jpg', '.jpeg', '.webp'),
) -> List[np.ndarray]:
    """Load images from *directory*, resize to TARGET_SIZE, return as float arrays."""
    path = Path(directory)
    if not path.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    files = sorted(
        f for f in path.iterdir() if f.suffix.lower() in extensions
    )[:n_samples]

    images: List[np.ndarray] = []
    for f in tqdm(files, desc=f"Loading {Path(directory).name}", unit="img"):
        try:
            img = Image.open(f).convert('RGB')
            img = resize_image(img, TARGET_SIZE)
            images.append(np.asarray(img, dtype=np.float64) / 255.0)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    return images


def jpeg_compress(image_array: np.ndarray, quality: int) -> np.ndarray:
    """JPEG-encode then decode *image_array* at the given *quality* factor.

    Args:
        image_array: float64 RGB image in [0, 1].
        quality: JPEG quality (1-100).

    Returns:
        Decoded float64 RGB image in [0, 1].
    """
    img_uint8 = (image_array * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='RGB')
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    decoded = Image.open(buf).convert('RGB')
    return np.asarray(decoded, dtype=np.float64) / 255.0


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(
    images: List[np.ndarray],
    extractor: PhaseExtractor,
    label: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ispc_values, cdpgc_values) for a list of images."""
    ispc_vals: List[float] = []
    cdpgc_vals: List[float] = []
    desc = f"Features [{label}]" if label else "Features"
    for img in tqdm(images, desc=desc, unit="img"):
        phase_maps = extractor.get_phase_maps(img)
        ispc_vals.append(float(np.mean(compute_ispc(phase_maps))))
        cdpgc_vals.append(float(np.mean(compute_cdpgc(phase_maps))))
    return np.array(ispc_vals), np.array(cdpgc_vals)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
GENERATOR_COLORS: Dict[str, str] = {
    'SD_v1.1': '#e41a1c',
    'SD_v1.5': '#ff7f00',
    'SD_v2.1': '#984ea3',
    'Kandinsky_2.1': '#4daf4a',
    'Midjourney_v4': '#a65628',
    'Midjourney_v5': '#f781bf',
    'Midjourney_v5.1': '#999999',
}


def plot_distribution(
    real_values: np.ndarray,
    fake_dict: Dict[str, np.ndarray],
    metric_name: str,
    cohens_d_dict: Dict[str, float],
    output_path: str,
) -> None:
    """Create a publication-quality KDE/histogram overlay plot for a metric."""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(real_values, ax=ax, color='#377eb8', linewidth=2.5,
                label='Real (FFHQ)', fill=True, alpha=0.25)

    for name, values in fake_dict.items():
        color = GENERATOR_COLORS.get(name, '#333333')
        d_val = cohens_d_dict.get(name, 0.0)
        sns.kdeplot(values, ax=ax, color=color, linewidth=1.8,
                    label=f"{name} (d={d_val:.2f})")

    ax.set_xlabel(metric_name, fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(f'{metric_name} Distribution: Real vs. Generators', fontsize=15)
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved distribution plot -> {output_path}")


def plot_cohens_d_summary(
    decision_table: List[Dict],
    output_path: str,
) -> None:
    """Bar chart of Cohen's d per generator with GO threshold line."""
    names = [r['generator'] for r in decision_table]
    d_vals = [r['d_combined'] for r in decision_table]
    colors = ['#4daf4a' if d > COHENS_D_CUTOFF else '#e41a1c' for d in d_vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, d_vals, color=colors, edgecolor='black', linewidth=0.6)
    ax.axhline(y=COHENS_D_CUTOFF, color='black', linestyle='--', linewidth=1.2,
               label=f'GO threshold (d={COHENS_D_CUTOFF})')

    for bar, d in zip(bars, d_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{d:.2f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel("Cohen's d", fontsize=13)
    ax.set_title("Cohen's d per Generator (max of ISPC, CDPGC)", fontsize=15)
    ax.legend(fontsize=11)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved Cohen's d summary plot -> {output_path}")


def plot_compression_robustness(
    compression_results: Dict[int, Dict[str, Dict[str, float]]],
    output_path: str,
) -> None:
    """Line plot: x = JPEG quality, y = Cohen's d per generator."""
    quality_factors = sorted(compression_results.keys(), reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for metric_key, metric_label, ax in zip(
        ['d_ispc', 'd_cdpgc'], ['ISPC', 'CDPGC'], axes
    ):
        for gen_name in compression_results[quality_factors[0]]:
            color = GENERATOR_COLORS.get(gen_name, '#333333')
            y_vals = [compression_results[qf][gen_name][metric_key] for qf in quality_factors]
            ax.plot(quality_factors, y_vals, marker='o', linewidth=1.8,
                    color=color, label=gen_name)

        ax.axhline(y=COHENS_D_CUTOFF, color='black', linestyle='--',
                    linewidth=1.0, label=f'd={COHENS_D_CUTOFF}')
        ax.set_xlabel('JPEG Quality', fontsize=12)
        ax.set_ylabel("Cohen's d", fontsize=12)
        ax.set_title(f'{metric_label} Compression Robustness', fontsize=14)
        ax.legend(fontsize=8, loc='best')
        ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved compression robustness plot -> {output_path}")


# ---------------------------------------------------------------------------
# Compression test
# ---------------------------------------------------------------------------
def run_compression_test(
    real_images: List[np.ndarray],
    fake_images_dict: Dict[str, List[np.ndarray]],
    extractor: PhaseExtractor,
    quality_factors: List[int] = [95, 85, 75, 50],
    output_dir: str = 'results/pilot_study',
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Run JPEG compression robustness test.

    For each quality factor, compress images and recompute Cohen's d.

    Returns:
        Nested dict: quality_factor -> generator_name -> {d_ispc, d_cdpgc}.
    """
    compression_results: Dict[int, Dict[str, Dict[str, float]]] = {}

    for qf in quality_factors:
        logger.info(f"\n--- JPEG quality={qf} ---")

        # Compress real images
        real_compressed = [jpeg_compress(img, qf) for img in
                           tqdm(real_images, desc=f"Compress real Q={qf}", unit="img")]
        real_ispc, real_cdpgc = extract_features(
            real_compressed, extractor, label=f"real Q={qf}"
        )

        qf_results: Dict[str, Dict[str, float]] = {}
        for gen_name, fake_imgs in fake_images_dict.items():
            fake_compressed = [jpeg_compress(img, qf) for img in
                               tqdm(fake_imgs, desc=f"Compress {gen_name} Q={qf}", unit="img")]
            fake_ispc, fake_cdpgc = extract_features(
                fake_compressed, extractor, label=f"{gen_name} Q={qf}"
            )

            d_ispc = cohens_d(real_ispc, fake_ispc)
            d_cdpgc = cohens_d(real_cdpgc, fake_cdpgc)
            qf_results[gen_name] = {'d_ispc': d_ispc, 'd_cdpgc': d_cdpgc}
            logger.info(f"  {gen_name}: d_ispc={d_ispc:.3f}, d_cdpgc={d_cdpgc:.3f}")

        compression_results[qf] = qf_results

    # Plot
    plot_path = os.path.join(output_dir, 'compression_robustness.png')
    plot_compression_robustness(compression_results, plot_path)

    return compression_results


# ---------------------------------------------------------------------------
# Email notification
# ---------------------------------------------------------------------------
def send_email_notification(
    final_decision: str,
    decision_table: List[Dict],
    go_count: int,
    total_generators: int,
    compression_ran: bool,
) -> None:
    """Send an email with the pilot study GO/NO-GO result."""
    subject = f"[PhaseForensics] Pilot Study Result: {final_decision}"

    body_lines = [
        "PhaseForensics Pilot Study Complete",
        "=" * 50,
        f"Timestamp: {datetime.now().isoformat()}",
        f"Final Decision: {final_decision}",
        f"Generators passing (d > {COHENS_D_CUTOFF}): {go_count}/{total_generators}",
        "",
        "Decision Table:",
        "-" * 50,
        f"{'Generator':<20s} {'d_ispc':>8s} {'d_cdpgc':>8s} {'d_max':>8s} {'Status':>12s}",
        "-" * 50,
    ]
    for row in decision_table:
        body_lines.append(
            f"{row['generator']:<20s} {row['d_ispc']:>8.3f} {row['d_cdpgc']:>8.3f} "
            f"{row['d_combined']:>8.3f} {row['status']:>12s}"
        )
    body_lines.append("-" * 50)
    if compression_ran:
        body_lines.append("JPEG compression robustness test was also executed.")
    body_lines.append("")
    body_lines.append("-- PhaseForensics automated notification")

    body = "\n".join(body_lines)

    msg = MIMEMultipart()
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_FROM
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_FROM, msg.as_string())
        server.quit()
        logger.info("Email notification sent successfully.")
    except Exception as e:
        logger.warning(f"Failed to send email notification: {e}")


# ---------------------------------------------------------------------------
# Main pilot study
# ---------------------------------------------------------------------------
def run_pilot_study(
    data_config: Dict[str, str],
    n_samples: int = 100,
    output_dir: str = 'results/pilot_study',
    skip_compression: bool = False,
    config_path: Optional[str] = None,
) -> None:
    """Run the pilot study.

    Args:
        data_config: Dict mapping generator names to image directories.
        n_samples: Number of images per generator.
        output_dir: Where to save results.
        skip_compression: If True, skip the JPEG compression test.
        config_path: Optional path to YAML config file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load YAML config if provided
    yaml_config: Dict = {}
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f) or {}
        logger.info(f"Loaded config from {config_path}")

    nlevels = yaml_config.get('dtcwt', {}).get('nlevels', 5)
    extractor = PhaseExtractor(nlevels=nlevels)

    results: Dict[str, Dict] = {}
    raw_images: Dict[str, List[np.ndarray]] = {}

    for name, directory in data_config.items():
        logger.info(f"Processing {name}...")
        images = load_images_from_dir(directory, n_samples)

        if not images:
            logger.warning(f"  No images found for {name}, skipping")
            continue

        raw_images[name] = images
        ispc_values, cdpgc_values = extract_features(images, extractor, label=name)

        results[name] = {
            'ispc_values': ispc_values.tolist(),
            'cdpgc_values': cdpgc_values.tolist(),
            'ispc_mean': float(np.mean(ispc_values)),
            'ispc_std': float(np.std(ispc_values)),
            'cdpgc_mean': float(np.mean(cdpgc_values)),
            'cdpgc_std': float(np.std(cdpgc_values)),
            'n_images': len(images),
        }
        logger.info(
            f"  {name}: ISPC={results[name]['ispc_mean']:.4f}+/-{results[name]['ispc_std']:.4f}, "
            f"CDPGC={results[name]['cdpgc_mean']:.6f}+/-{results[name]['cdpgc_std']:.6f}"
        )

    # --- Compute Cohen's d for each generator vs real -----------------------
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
    decision_table: List[Dict] = []

    # Collect per-generator distributions for plotting
    fake_ispc_dict: Dict[str, np.ndarray] = {}
    fake_cdpgc_dict: Dict[str, np.ndarray] = {}
    cohens_d_ispc_dict: Dict[str, float] = {}
    cohens_d_cdpgc_dict: Dict[str, float] = {}

    for name, data in results.items():
        if name == 'real':
            continue
        total_generators += 1

        fake_ispc = np.array(data['ispc_values'])
        fake_cdpgc = np.array(data['cdpgc_values'])

        d_ispc = cohens_d(real_ispc, fake_ispc)
        d_cdpgc = cohens_d(real_cdpgc, fake_cdpgc)
        d_combined = max(d_ispc, d_cdpgc)

        if d_combined > COHENS_D_CUTOFF:
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

        fake_ispc_dict[name] = fake_ispc
        fake_cdpgc_dict[name] = fake_cdpgc
        cohens_d_ispc_dict[name] = d_ispc
        cohens_d_cdpgc_dict[name] = d_cdpgc

        logger.info(
            f"  {name:20s}: d_ispc={d_ispc:.3f}, d_cdpgc={d_cdpgc:.3f}, "
            f"d_max={d_combined:.3f} [{status}]"
        )

    # --- Final decision ------------------------------------------------------
    logger.info("\n" + "-" * 60)

    if go_count >= GO_THRESHOLD:
        final = "GO"
        logger.info(
            f"FINAL DECISION: *** GO *** ({go_count}/{total_generators} generators pass)"
        )
    elif go_count >= CONDITIONAL_THRESHOLD:
        final = "CONDITIONAL GO"
        logger.info(f"FINAL DECISION: CONDITIONAL GO ({go_count}/{total_generators})")
        logger.info("  -> Consider amplitude+phase combination")
    else:
        final = "NO-GO"
        logger.info(f"FINAL DECISION: NO-GO ({go_count}/{total_generators})")
        logger.info("  -> Switch to FoMFP or FAP")

    # --- Visualization -------------------------------------------------------
    logger.info("\nGenerating visualizations...")

    plot_distribution(
        real_ispc, fake_ispc_dict, 'ISPC', cohens_d_ispc_dict,
        os.path.join(output_dir, 'ispc_distributions.png'),
    )
    plot_distribution(
        real_cdpgc, fake_cdpgc_dict, 'CDPGC', cohens_d_cdpgc_dict,
        os.path.join(output_dir, 'cdpgc_distributions.png'),
    )
    plot_cohens_d_summary(
        decision_table,
        os.path.join(output_dir, 'cohens_d_summary.png'),
    )

    # --- JPEG compression test -----------------------------------------------
    compression_ran = False
    if not skip_compression and raw_images.get('real'):
        logger.info("\nRunning JPEG compression robustness test...")
        fake_images_for_comp = {
            name: raw_images[name]
            for name in fake_ispc_dict
            if name in raw_images
        }
        run_compression_test(
            real_images=raw_images['real'],
            fake_images_dict=fake_images_for_comp,
            extractor=extractor,
            output_dir=output_dir,
        )
        compression_ran = True

    # --- Save JSON results ---------------------------------------------------
    output = {
        'timestamp': datetime.now().isoformat(),
        'n_samples_per_generator': n_samples,
        'decision': final,
        'go_count': go_count,
        'total_generators': total_generators,
        'go_threshold': f">= {GO_THRESHOLD}/{TOTAL_GENERATORS}",
        'conditional_threshold': f"{CONDITIONAL_THRESHOLD}-{GO_THRESHOLD - 1}/{TOTAL_GENERATORS}",
        'decision_table': decision_table,
        'stats': {
            k: {kk: vv for kk, vv in v.items()
                 if kk not in ('ispc_values', 'cdpgc_values')}
            for k, v in results.items()
        },
    }

    output_path = os.path.join(output_dir, 'pilot_study_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # --- Email notification --------------------------------------------------
    send_email_notification(
        final_decision=final,
        decision_table=decision_table,
        go_count=go_count,
        total_generators=total_generators,
        compression_ran=compression_ran,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='PhaseForensics Pilot Study: GO/NO-GO decision'
    )
    parser.add_argument(
        '--n-samples', type=int, default=100,
        help='Number of images to sample per generator (default: 100)',
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/pilot_study',
        help='Directory to save results and plots (default: results/pilot_study)',
    )
    parser.add_argument(
        '--config', type=str, default='configs/default.yaml',
        help='Path to YAML config file (default: configs/default.yaml)',
    )
    parser.add_argument(
        '--skip-compression', action='store_true',
        help='Skip the JPEG compression robustness test',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data_config: Dict[str, str] = {
        'real': REAL_DIR,
    }
    data_config.update(GENERATOR_MAP)

    # Filter to existing directories only
    existing = {k: v for k, v in data_config.items() if Path(v).exists()}

    if len(existing) < 2:
        logger.error(
            "Need at least 'real' + 1 generator directory. "
            "Check that dataset paths exist."
        )
        logger.info("Expected paths:")
        for name, path in data_config.items():
            status = "OK" if Path(path).exists() else "MISSING"
            logger.info(f"  [{status}] {name}: {path}")
    else:
        missing = set(data_config.keys()) - set(existing.keys())
        if missing:
            logger.warning(f"Missing directories for: {missing}")

        run_pilot_study(
            data_config=existing,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            skip_compression=args.skip_compression,
            config_path=args.config,
        )
