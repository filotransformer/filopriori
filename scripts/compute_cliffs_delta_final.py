#!/usr/bin/env python3
"""
Compute Cliff's delta from the paper's reported per-project APFD values
(Table III in the IEEE TSE paper) and from the per-build CSVs for industrial.

This ensures we compute effect sizes on the EXACT same data used in the paper.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def cliffs_delta(x, y):
    """Cliff's delta: positive means x > y on average."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    more = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    delta = (more - less) / (n1 * n2)
    abs_d = abs(delta)
    if abs_d < 0.147:
        mag = "negligible"
    elif abs_d < 0.33:
        mag = "small"
    elif abs_d < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return delta, mag


# =============================================================================
# RTPTORRENT: Per-project APFD from Table III (exact paper values, 20 projects)
# =============================================================================
projects = [
    "facebook/buck", "apache/sling", "eclipse/jetty.project", "jOOQ/jOOQ",
    "julianhyde/optiq", "jcabi/jcabi-github", "CloudifySource/cloudify",
    "square/okhttp", "Graylog2/graylog2-server", "doanduyhai/Achilles",
    "SonarSource/sonarqube", "thinkaurelius/titan", "deeplearning4j/dl4j",
    "jsprit/jsprit", "dynjs/dynjs", "adamfisk/LittleProxy",
    "neuland/jade4j", "brettwooldridge/HikariCP", "DSpace/DSpace",
    "l0rdn1kk0n/wicket-bootstrap"
]

fp = [0.9831, 0.9710, 0.9719, 0.9392, 0.9135, 0.9370, 0.9086, 0.8916,
      0.8940, 0.8693, 0.8671, 0.8410, 0.8361, 0.9426, 0.8001, 0.7196,
      0.7239, 0.6616, 0.6841, 0.5971]

tcpnet = [0.9149, 0.9589, 0.9673, 0.9311, 0.9122, 0.8431, 0.9096, 0.8679,
          0.8585, 0.8242, 0.8371, 0.8394, 0.8369, 0.9586, 0.6961, 0.7302,
          0.7285, 0.6497, 0.6425, 0.5993]

failrankbb = [0.9125, 0.9427, 0.9487, 0.9250, 0.8987, 0.8704, 0.8731, 0.8680,
              0.7708, 0.7739, 0.6934, 0.8350, 0.8428, 0.8959, 0.8294, 0.7807,
              0.7141, 0.7520, 0.7027, 0.6056]

deeporder = [0.8683, 0.9320, 0.9463, 0.9262, 0.9208, 0.7809, 0.8994, 0.8258,
             0.8817, 0.8135, 0.7964, 0.8441, 0.8330, 0.9305, 0.7777, 0.6718,
             0.7307, 0.6687, 0.6353, 0.5896]

noderank = [0.9658, 0.9410, 0.9265, 0.9267, 0.8872, 0.8515, 0.7288, 0.8341,
            0.8923, 0.8225, 0.8057, 0.8304, 0.7943, 0.8790, 0.7582, 0.6966,
            0.6715, 0.6151, 0.6454, 0.6026]

retecs = [0.7437, 0.8385, 0.9103, 0.7755, 0.8321, 0.6513, 0.7903, 0.7909,
          0.4312, 0.7040, 0.7680, 0.7371, 0.7480, 0.6215, 0.4948, 0.5742,
          0.5901, 0.5685, 0.4065, 0.6056]

print("=" * 70)
print("RTPTORRENT — Cliff's Delta (per-project means, n=20)")
print("  (Computed from Table III values in the paper)")
print("=" * 70)
print(f"\n{'Method':<15} {'APFD':>8} {'Cliff δ':>10} {'|δ|':>8} {'Magnitude':>12}")
print("-" * 56)

rtp_results = {}
for name, bl in [("TCP-Net", tcpnet), ("FailRank-BB", failrankbb),
                  ("DeepOrder", deeporder), ("NodeRank", noderank),
                  ("RETECS", retecs)]:
    delta, mag = cliffs_delta(fp, bl)
    mean_bl = np.mean(bl)
    rtp_results[name] = (delta, mag)
    print(f"{name:<15} {mean_bl:>8.4f} {delta:>+10.4f} {abs(delta):>8.4f} {mag:>12}")

# =============================================================================
# INDUSTRIAL: Per-build APFD from CSVs (n=277)
# =============================================================================
print("\n" + "=" * 70)
print("INDUSTRIAL — Cliff's Delta (per-build, n=277)")
print("  (Computed from per-build APFD CSVs)")
print("=" * 70)

base = Path(__file__).resolve().parent.parent / "results"

fp_df = pd.read_csv(base / "experiment_industry_optimized_v3" / "apfd_per_build_FULL_testcsv.csv")
fp_ind = fp_df.set_index('build_id')['apfd']

baselines_ind = {
    "DeepOrder": base / "deeporder_industry",
    "TCP-Net": base / "tcpnet_industry",
    "NodeRank": base / "noderank_industry",
    "RETECS": base / "retecs_industry",
    "FailRank-BB": base / "failrank_bb_industry",
}

print(f"\nFilo-Priori: n={len(fp_ind)}, mean={fp_ind.mean():.4f}")
print(f"\n{'Method':<15} {'APFD':>8} {'Cliff δ':>10} {'|δ|':>8} {'Magnitude':>12} {'n':>6}")
print("-" * 62)

ind_results = {}
for name, path in baselines_ind.items():
    csv_path = path / "apfd_per_build_FULL_testcsv.csv"
    if not csv_path.exists():
        print(f"{name:<15} N/A")
        continue
    bl_df = pd.read_csv(csv_path)
    bl = bl_df.set_index('build_id')['apfd']
    common = fp_ind.index.intersection(bl.index)
    delta, mag = cliffs_delta(fp_ind.loc[common].values, bl.loc[common].values)
    ind_results[name] = (delta, mag)
    print(f"{name:<15} {bl.mean():>8.4f} {delta:>+10.4f} {abs(delta):>8.4f} {mag:>12} {len(common):>6}")

# =============================================================================
# LaTeX output for paper tables
# =============================================================================
print("\n" + "=" * 70)
print("LATEX SNIPPETS FOR PAPER")
print("=" * 70)

print("\n% Industrial Dataset (Table II) — add δ and Effect columns")
print("% Method & APFD & Std & p-value & Cliff's δ & Effect & Δ vs FP")
for name in ["DeepOrder", "TCP-Net", "NodeRank", "RETECS", "FailRank-BB"]:
    if name in ind_results:
        d, m = ind_results[name]
        print(f"% {name}: δ = {d:+.3f} ({m})")

print("\n% RTPTorrent Dataset (Table III) — add δ and Effect columns")
print("% Method & APFD & Std & N & p-value & Cliff's δ & Effect & Δ vs FP")
for name in ["TCP-Net", "FailRank-BB", "DeepOrder", "NodeRank", "RETECS"]:
    if name in rtp_results:
        d, m = rtp_results[name]
        print(f"% {name}: δ = {d:+.3f} ({m})")


if __name__ == "__main__":
    main()
