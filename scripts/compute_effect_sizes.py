#!/usr/bin/env python3
"""
Compute Cliff's delta effect sizes for Filo-Priori vs all baselines.
Used to populate Tables II and III in the IEEE TSE paper.

Cliff's delta interpretation (Romano et al., 2006):
  |d| < 0.147  → negligible
  |d| < 0.33   → small
  |d| < 0.474  → medium
  |d| >= 0.474 → large
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def cliffs_delta(x, y):
    """
    Compute Cliff's delta between two samples.

    Cliff's delta = (# concordant pairs - # discordant pairs) / (n1 * n2)

    Returns:
        delta: float in [-1, 1]. Positive means x tends to be larger than y.
        magnitude: str interpretation
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Remove NaN
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan, "n/a"

    # Count concordant and discordant pairs
    more = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                more += 1
            elif xi < yj:
                less += 1

    delta = (more - less) / (n1 * n2)

    # Interpret magnitude (Romano et al., 2006)
    abs_d = abs(delta)
    if abs_d < 0.147:
        magnitude = "negligible"
    elif abs_d < 0.33:
        magnitude = "small"
    elif abs_d < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"

    return delta, magnitude


def load_industrial_apfd(results_dir, method_name):
    """Load per-build APFD for an industrial method."""
    path = results_dir / "apfd_per_build_FULL_testcsv.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return None
    df = pd.read_csv(path)
    return df.set_index('build_id')['apfd']


def load_rtptorrent_per_project(results_dir):
    """Load per-project mean APFD for RTPTorrent."""
    path = results_dir / "per_project_apfd.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return None
    df = pd.read_csv(path)
    if 'project' in df.columns:
        return df.set_index('project')['mean_apfd']
    return None


def load_rtptorrent_per_build(results_dir):
    """Load per-build APFD for RTPTorrent."""
    path = results_dir / "apfd_per_build_FULL_testcsv.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


def main():
    base = Path(__file__).resolve().parent.parent / "results"

    # =========================================================================
    # INDUSTRIAL DATASET (277 builds)
    # =========================================================================
    print("=" * 70)
    print("INDUSTRIAL DATASET — Cliff's Delta (per-build, n=277)")
    print("=" * 70)

    fp_industry = load_industrial_apfd(base / "experiment_industry_optimized_v3", "Filo-Priori")

    industrial_baselines = {
        "DeepOrder": base / "deeporder_industry",
        "TCP-Net": base / "tcpnet_industry",
        "NodeRank": base / "noderank_industry",
        "RETECS": base / "retecs_industry",
        "FailRank-BB": base / "failrank_bb_industry",
    }

    print(f"\nFilo-Priori: n={len(fp_industry)}, mean APFD={fp_industry.mean():.4f}")
    print(f"\n{'Method':<15} {'Mean APFD':>10} {'Cliff δ':>10} {'Magnitude':>12} {'n_paired':>10}")
    print("-" * 60)

    for name, path in industrial_baselines.items():
        bl = load_industrial_apfd(path, name)
        if bl is None:
            print(f"{name:<15} {'N/A':>10}")
            continue

        # Align on common build_ids
        common = fp_industry.index.intersection(bl.index)
        fp_vals = fp_industry.loc[common].values
        bl_vals = bl.loc[common].values

        delta, mag = cliffs_delta(fp_vals, bl_vals)
        print(f"{name:<15} {bl.mean():>10.4f} {delta:>10.4f} {mag:>12} {len(common):>10}")

    # =========================================================================
    # RTPTORRENT — Per-Project (n=20, used for Wilcoxon in paper)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RTPTORRENT — Cliff's Delta (per-project means, n=20)")
    print("=" * 70)

    fp_rtp = load_rtptorrent_per_project(base / "filopriori_rtptorrent_v14")

    rtp_baselines = {
        "TCP-Net": base / "tcpnet_rtptorrent",
        "FailRank-BB": base / "failrank_bb_rtptorrent",
        "DeepOrder": base / "deeporder_rtptorrent",
        "NodeRank": base / "noderank_rtptorrent",
        "RETECS": base / "retecs_rtptorrent",
    }

    if fp_rtp is not None:
        print(f"\nFilo-Priori V14: n={len(fp_rtp)}, grand mean APFD={fp_rtp.mean():.4f}")
        print(f"\n{'Method':<15} {'Mean APFD':>10} {'Cliff δ':>10} {'Magnitude':>12} {'n_paired':>10}")
        print("-" * 60)

        for name, path in rtp_baselines.items():
            bl = load_rtptorrent_per_project(path)
            if bl is None:
                print(f"{name:<15} {'N/A':>10}")
                continue

            common = fp_rtp.index.intersection(bl.index)
            fp_vals = fp_rtp.loc[common].values
            bl_vals = bl.loc[common].values

            delta, mag = cliffs_delta(fp_vals, bl_vals)
            print(f"{name:<15} {bl.mean():>10.4f} {delta:>10.4f} {mag:>12} {len(common):>10}")

    # =========================================================================
    # RTPTORRENT — Per-Build (n=2937, supplementary)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RTPTORRENT — Cliff's Delta (per-build, n≈2937, supplementary)")
    print("=" * 70)

    fp_builds = load_rtptorrent_per_build(base / "filopriori_rtptorrent_v14")

    if fp_builds is not None:
        print(f"\nFilo-Priori V14: n_builds={len(fp_builds)}, mean APFD={fp_builds['apfd'].mean():.4f}")
        print(f"\n{'Method':<15} {'Mean APFD':>10} {'Cliff δ':>10} {'Magnitude':>12} {'n_paired':>10}")
        print("-" * 60)

        for name, path in rtp_baselines.items():
            bl_df = load_rtptorrent_per_build(path)
            if bl_df is None:
                print(f"{name:<15} {'N/A':>10}")
                continue

            # Align on (project, build_id)
            fp_key = fp_builds.set_index(['project', 'build_id'])['apfd']
            bl_key = bl_df.set_index(['project', 'build_id'])['apfd']
            common = fp_key.index.intersection(bl_key.index)

            fp_vals = fp_key.loc[common].values
            bl_vals = bl_key.loc[common].values

            delta, mag = cliffs_delta(fp_vals, bl_vals)
            print(f"{name:<15} {bl_df['apfd'].mean():>10.4f} {delta:>10.4f} {mag:>12} {len(common):>10}")


if __name__ == "__main__":
    main()
