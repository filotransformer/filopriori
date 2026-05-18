# Experiments — Filo-Priori vs Baselines

This directory contains the experiment scripts that produce the results reported in the paper.
See `CLAUDE.md` at the project root for frozen hyperparameters and parameter protection rules.

## Directory Structure

```
experiments/
├── README.md                              # This file
├── run_filopriori_rtptorrent_v15.py       # Filo-Priori V15 - RTPTorrent (APFD 0.8540)
├── run_deeporder_industry.py              # DeepOrder - Industry
├── run_deeporder_rtptorrent.py            # DeepOrder - RTPTorrent
├── run_noderank_industry.py               # NodeRank - Industry
├── run_noderank_rtptorrent.py             # NodeRank - RTPTorrent
├── run_retecs_industry.py                 # RETECS - Industry
├── run_retecs_rtptorrent.py               # RETECS - RTPTorrent
├── run_tcpnet_industry.py                 # TCP-Net - Industry
├── run_tcpnet_rtptorrent.py               # TCP-Net - RTPTorrent
├── run_failrank_bb_industry.py            # FailRank-BB - Industry
├── run_failrank_bb_rtptorrent.py          # FailRank-BB - RTPTorrent
└── archived/                              # Old/deprecated Filo-Priori versions (not in paper)
```

## Valid Experiments for the Paper

### Filo-Priori

| Dataset | Script | APFD | Command |
|---------|--------|------|---------|
| **Industrial QTA** | `main.py` (project root) | **0.7611** | `python main.py --config configs/experiment_industry_optimized_v3.yaml` |
| **RTPTorrent (20 projects)** | `experiments/run_filopriori_rtptorrent_v15.py` | **0.8540** | `python experiments/run_filopriori_rtptorrent_v15.py` |

### Baselines

```bash
# Activate virtual environment first
source venv/bin/activate

# --- Industry Dataset ---
python experiments/run_deeporder_industry.py     # DeepOrder (APFD 0.6890)
python experiments/run_noderank_industry.py       # NodeRank  (APFD 0.6609)
python experiments/run_retecs_industry.py         # RETECS    (APFD 0.6406)
python experiments/run_tcpnet_industry.py         # TCP-Net   (APFD 0.6704)
python experiments/run_failrank_bb_industry.py    # FailRank  (APFD 0.5953)

# --- RTPTorrent Dataset ---
python experiments/run_deeporder_rtptorrent.py    # DeepOrder (APFD 0.8136)
python experiments/run_noderank_rtptorrent.py     # NodeRank  (APFD 0.8038)
python experiments/run_retecs_rtptorrent.py       # RETECS    (APFD 0.6791)
python experiments/run_tcpnet_rtptorrent.py       # TCP-Net   (APFD 0.8253)
python experiments/run_failrank_bb_rtptorrent.py  # FailRank  (APFD 0.8218)
```

## Results — Industry Dataset (277 Builds with Failures)

| Method | APFD | Std | vs Filo-Priori |
|--------|------|-----|----------------|
| **Filo-Priori** | **0.7611** | 0.189 | -- |
| DeepOrder | 0.6890 | 0.266 | +10.2% |
| TCP-Net | 0.6704 | 0.271 | +13.3% |
| NodeRank | 0.6609 | 0.270 | +14.9% |
| RETECS | 0.6406 | 0.281 | +18.6% |
| FailRank-BB | 0.5953 | 0.263 | +27.6% |

## Results — RTPTorrent (20 Projects, 2,937 Builds with Failures)

| Method | Grand Mean APFD | Std | Builds | Time |
|--------|----------------|-----|--------|------|
| **Filo-Priori V15** | **0.8540** | 0.112 | 2,937 | ~3.1h |
| TCP-Net | 0.8253 | 0.110 | 2,937 | ~12h |
| FailRank-BB | 0.8218 | 0.092 | 2,937 | -- |
| DeepOrder | 0.8136 | 0.104 | 2,937 | ~21h |
| NodeRank | 0.8038 | 0.109 | 2,937 | ~21h |
| RETECS | 0.6791 | 0.156 | 2,937 | ~1.4h |

## Scientific Comparability Guarantees

All experiments are designed for fair comparison:

1. **Same Data Splits**
   - Industry: `datasets/01_industry/train.csv` / `test.csv`
   - RTPTorrent: Temporal 80% train / 20% test per project

2. **Same APFD Metric**
   - Formula: `APFD = 1 - sum(rank_failures) / (n_failures * n_tests) + 1 / (2 * n_tests)`
   - Calculated per build
   - Edge case: builds with 1 test case = APFD 1.0

3. **Same Inclusion Criteria**
   - Only builds with at least 1 failure
   - Same number of evaluated builds

4. **Statistical Tests**
   - Wilcoxon signed-rank (paired, non-parametric)
   - Effect size: Cliff's delta and Cohen's d
   - Bootstrap confidence intervals

## Output Format

Each experiment generates files in `results/<method>_<dataset>/`:

| File | Description |
|------|-------------|
| `apfd_per_build_FULL_testcsv.csv` | APFD per build (standard format) |
| `per_project_apfd.csv` | Mean APFD per project (RTPTorrent only) |
| `aggregate_results.json` | Aggregated results with per-project details |
| `experiment_summary.json` | Experiment summary and configuration |
| `comparison_summary.txt` | Human-readable summary |

Standard CSV format:
```csv
method_name,build_id,test_scenario,count_tc,count_commits,apfd,time
```

## Archived Experiments

Old Filo-Priori versions (v3, v9, v10, v11, v12, v13, and original) are in `archived/`.
These were intermediate development versions and are **NOT** used in the paper.
Only `run_filopriori_rtptorrent_v15.py` produces the published results.

---

*Last Updated: March 2026*
