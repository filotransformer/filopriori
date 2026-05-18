#!/usr/bin/env python3
"""
Edge-Type Ablation Study for the Industrial QTA Dataset

Runs incremental multi-edge graph ablation on the industrial dataset (277 builds),
removing one edge type at a time from the full 5-edge configuration to measure
the individual contribution of each edge type to APFD.

Ablation variants:
  1. full            - All 5 edge types (co_failure + co_success + semantic + temporal + component)
  2. wo_cosuccess    - Remove co_success edges
  3. wo_component    - Remove component edges
  4. wo_semantic     - Remove semantic edges
  5. wo_temporal     - Remove temporal edges
  6. cofailure_only  - Co-failure only (remove all 4 additional edge types)

Each variant creates a modified YAML config and runs the full main.py pipeline.
Results are collected from apfd_per_build_FULL_testcsv.csv in each output directory.

Usage:
    # Run all ablations
    python experiments/run_edge_type_ablation_industry.py --all

    # Run a single ablation
    python experiments/run_edge_type_ablation_industry.py --variant wo_cosuccess

    # Run multiple specific ablations
    python experiments/run_edge_type_ablation_industry.py --variant wo_cosuccess wo_semantic

    # Only collect/display results (no training)
    python experiments/run_edge_type_ablation_industry.py --collect-only

    # Force re-run even if results exist
    python experiments/run_edge_type_ablation_industry.py --all --force
"""

import argparse
import copy
import csv
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_industry_optimized_v3.yaml"
ABLATION_CONFIG_DIR = PROJECT_ROOT / "configs" / "ablation"
RESULTS_BASE_DIR = PROJECT_ROOT / "results" / "edge_type_ablation_industry"
MAIN_SCRIPT = PROJECT_ROOT / "main.py"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Full set of edge types in the industrial multi-edge graph
# ---------------------------------------------------------------------------
ALL_EDGE_TYPES = ["co_failure", "co_success", "semantic", "temporal", "component"]

# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------
VARIANTS: Dict[str, Dict] = {
    "full": {
        "label": "Full (all 5 edge types)",
        "edge_types": list(ALL_EDGE_TYPES),
        "use_multi_edge": True,
    },
    "wo_cosuccess": {
        "label": "w/o Co-Success",
        "edge_types": [e for e in ALL_EDGE_TYPES if e != "co_success"],
        "use_multi_edge": True,
    },
    "wo_component": {
        "label": "w/o Component",
        "edge_types": [e for e in ALL_EDGE_TYPES if e != "component"],
        "use_multi_edge": True,
    },
    "wo_semantic": {
        "label": "w/o Semantic",
        "edge_types": [e for e in ALL_EDGE_TYPES if e != "semantic"],
        "use_multi_edge": True,
    },
    "wo_temporal": {
        "label": "w/o Temporal",
        "edge_types": [e for e in ALL_EDGE_TYPES if e != "temporal"],
        "use_multi_edge": True,
    },
    "cofailure_only": {
        "label": "Co-failure only",
        "edge_types": ["co_failure"],
        "use_multi_edge": True,
    },
}

# Ordered list for presentation (full first, then single removals, then minimal)
VARIANT_ORDER = [
    "full",
    "wo_cosuccess",
    "wo_component",
    "wo_semantic",
    "wo_temporal",
    "cofailure_only",
]


# ============================================================================
# Helpers
# ============================================================================

def load_base_config() -> Dict:
    """Load the frozen industrial YAML config."""
    if not BASE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Base config not found: {BASE_CONFIG_PATH}")
    with open(BASE_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def create_variant_config(base_config: Dict, variant_name: str) -> Dict:
    """
    Create a modified config for a given ablation variant.

    Only the graph.edge_types, graph.use_multi_edge, experiment.name,
    output.results_dir, and graph.cache_path are changed. All model
    hyperparameters remain frozen.
    """
    variant = VARIANTS[variant_name]
    cfg = copy.deepcopy(base_config)

    # Experiment metadata
    cfg["experiment"]["name"] = f"edge_ablation_{variant_name}"
    cfg["experiment"]["description"] = (
        f"Edge-type ablation: {variant['label']}"
    )

    # Graph edge types
    cfg["graph"]["edge_types"] = list(variant["edge_types"])
    cfg["graph"]["use_multi_edge"] = variant["use_multi_edge"]

    # Use a separate graph cache per variant to avoid stale cached graphs
    cfg["graph"]["cache_path"] = str(
        PROJECT_ROOT / "cache" / "01_industry" / f"multi_edge_graph_{variant_name}.pkl"
    )

    # Output directory
    cfg["output"]["results_dir"] = str(
        RESULTS_BASE_DIR / variant_name
    )

    return cfg


def save_variant_config(cfg: Dict, variant_name: str) -> Path:
    """Write the variant config YAML to disk and return its path."""
    ABLATION_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = ABLATION_CONFIG_DIR / f"edge_ablation_{variant_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    logger.info(f"  Config written: {config_path}")
    return config_path


def variant_is_complete(variant_name: str) -> bool:
    """Check whether this variant already has a full APFD results file."""
    apfd_path = RESULTS_BASE_DIR / variant_name / "apfd_per_build_FULL_testcsv.csv"
    if not apfd_path.exists():
        return False
    # Verify the file has content (more than just a header)
    try:
        with open(apfd_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            first_row = next(reader, None)
            return first_row is not None
    except Exception:
        return False


def run_variant(variant_name: str, force: bool = False) -> bool:
    """
    Run main.py for a single ablation variant.

    Returns True if the run completed (or was already complete).
    """
    label = VARIANTS[variant_name]["label"]

    if not force and variant_is_complete(variant_name):
        logger.info(f"  [{variant_name}] Already complete, skipping. Use --force to re-run.")
        return True

    logger.info(f"  [{variant_name}] Preparing config for: {label}")
    base_config = load_base_config()
    cfg = create_variant_config(base_config, variant_name)
    config_path = save_variant_config(cfg, variant_name)

    # Ensure results directory exists
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run main.py as a subprocess so each variant gets a clean process.
    # Stream output live to both console and log file so the user can
    # see progress (each variant takes ~30-45 minutes).
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--config", str(config_path),
    ]
    logger.info(f"  [{variant_name}] Running: {' '.join(cmd)}")
    logger.info(f"  [{variant_name}] NOTE: Each variant takes ~30-45 min. Output streams below.")

    log_path = results_dir / "run_log.txt"
    t0 = time.time()
    try:
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line-buffered
            )
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
            process.wait()

        elapsed = time.time() - t0

        if process.returncode != 0:
            logger.error(
                f"  [{variant_name}] FAILED (exit code {process.returncode}) "
                f"after {elapsed:.0f}s. See {log_path}"
            )
            return False

        logger.info(f"  [{variant_name}] Completed in {elapsed:.0f}s")
        return True

    except Exception as e:
        logger.error(f"  [{variant_name}] ERROR: {e}")
        return False


def read_apfd(variant_name: str) -> Optional[Dict]:
    """
    Read APFD results for a variant from its apfd_per_build CSV.

    Returns a dict with mean_apfd, median_apfd, std_apfd, num_builds,
    or None if results are not available.
    """
    # Prefer the FULL test.csv results (277 builds)
    apfd_path = RESULTS_BASE_DIR / variant_name / "apfd_per_build_FULL_testcsv.csv"
    if not apfd_path.exists():
        # Fall back to the split-only results
        apfd_path = RESULTS_BASE_DIR / variant_name / "apfd_per_build.csv"
    if not apfd_path.exists():
        return None

    try:
        import pandas as pd

        df = pd.read_csv(apfd_path)
        if "apfd" not in df.columns:
            logger.warning(f"  [{variant_name}] No 'apfd' column in {apfd_path}")
            return None

        apfd_values = df["apfd"].dropna()
        if len(apfd_values) == 0:
            return None

        return {
            "mean_apfd": float(apfd_values.mean()),
            "median_apfd": float(apfd_values.median()),
            "std_apfd": float(apfd_values.std()),
            "num_builds": int(len(apfd_values)),
        }
    except Exception as e:
        logger.warning(f"  [{variant_name}] Error reading APFD: {e}")
        return None


def collect_and_display_results():
    """Collect APFD from all variants and print a comparison table."""
    logger.info("")
    logger.info("=" * 78)
    logger.info("EDGE-TYPE ABLATION RESULTS (Industrial QTA, 277 builds)")
    logger.info("=" * 78)

    results = {}
    full_apfd = None

    for vname in VARIANT_ORDER:
        r = read_apfd(vname)
        if r is not None:
            results[vname] = r
            if vname == "full":
                full_apfd = r["mean_apfd"]

    if not results:
        logger.warning("No results found. Run ablations first with --all.")
        return

    # Table header
    header = f"{'Variant':<30s} {'Edge Types':>10s} {'APFD':>8s} {'Delta':>8s} {'Builds':>7s}"
    logger.info(header)
    logger.info("-" * 78)

    for vname in VARIANT_ORDER:
        if vname not in results:
            continue
        r = results[vname]
        label = VARIANTS[vname]["label"]
        n_edges = len(VARIANTS[vname]["edge_types"])

        if full_apfd is not None and vname != "full":
            delta = r["mean_apfd"] - full_apfd
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "--"

        row = (
            f"{label:<30s} {n_edges:>10d} "
            f"{r['mean_apfd']:>8.4f} {delta_str:>8s} {r['num_builds']:>7d}"
        )
        logger.info(row)

    logger.info("-" * 78)

    # Save results to JSON for downstream use
    summary_path = RESULTS_BASE_DIR / "summary.json"
    import json
    summary = {}
    for vname in VARIANT_ORDER:
        if vname in results:
            summary[vname] = {
                "label": VARIANTS[vname]["label"],
                "edge_types": VARIANTS[vname]["edge_types"],
                **results[vname],
            }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")

    # Save results to CSV
    csv_path = RESULTS_BASE_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "label", "edge_types", "num_edge_types",
                         "mean_apfd", "median_apfd", "std_apfd", "num_builds", "delta_vs_full"])
        for vname in VARIANT_ORDER:
            if vname not in results:
                continue
            r = results[vname]
            v = VARIANTS[vname]
            delta = (r["mean_apfd"] - full_apfd) if full_apfd is not None and vname != "full" else 0.0
            writer.writerow([
                vname,
                v["label"],
                "+".join(v["edge_types"]),
                len(v["edge_types"]),
                f"{r['mean_apfd']:.4f}",
                f"{r['median_apfd']:.4f}",
                f"{r['std_apfd']:.4f}",
                r["num_builds"],
                f"{delta:+.4f}",
            ])
    logger.info(f"CSV saved to {csv_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Edge-type ablation study for the industrial dataset (277 builds)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all ablation variants",
    )
    parser.add_argument(
        "--variant", nargs="+", choices=list(VARIANTS.keys()),
        help="Run specific ablation variant(s)",
    )
    parser.add_argument(
        "--collect-only", action="store_true",
        help="Only collect and display results (no training)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run even if results already exist",
    )
    args = parser.parse_args()

    # Determine which variants to run
    if args.collect_only:
        variants_to_run = []
    elif args.all:
        variants_to_run = list(VARIANT_ORDER)
    elif args.variant:
        variants_to_run = args.variant
    else:
        parser.print_help()
        print("\nError: specify --all, --variant <name>, or --collect-only")
        sys.exit(1)

    # Ensure base directories exist
    RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Run variants
    if variants_to_run:
        logger.info("=" * 78)
        logger.info("EDGE-TYPE ABLATION STUDY -- Industrial QTA Dataset")
        logger.info("=" * 78)
        logger.info(f"Variants to run: {variants_to_run}")
        logger.info(f"Force re-run: {args.force}")
        logger.info(f"Results directory: {RESULTS_BASE_DIR}")
        logger.info("")

        success_count = 0
        fail_count = 0

        for vname in variants_to_run:
            logger.info(f"\n{'='*78}")
            logger.info(f"VARIANT: {VARIANTS[vname]['label']}")
            logger.info(f"Edge types: {VARIANTS[vname]['edge_types']}")
            logger.info(f"{'='*78}")

            ok = run_variant(vname, force=args.force)
            if ok:
                success_count += 1
            else:
                fail_count += 1

        logger.info("")
        logger.info(f"Completed: {success_count} succeeded, {fail_count} failed")

    # Always collect and display results at the end
    collect_and_display_results()


if __name__ == "__main__":
    main()
