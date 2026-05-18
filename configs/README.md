# Configurations

**Last Updated:** March 2026

---

## Valid Configurations for the Paper

| Config | Dataset | APFD | Status |
|--------|---------|------|--------|
| **`experiment_industry_optimized_v3.yaml`** | Industrial QTA | **0.7611** | FROZEN |

For RTPTorrent (APFD 0.8540), the configuration is embedded directly in:
`experiments/run_filopriori_rtptorrent_v15.py`

**WARNING:** Do not modify these configurations. See `CLAUDE.md` at the project root.

---

## Other Config Files (Reference Only)

| File | Purpose | Notes |
|------|---------|-------|
| `experiment.yaml` | Default/base config | Not used in paper |
| `experiment_rtptorrent.yaml` | RTPTorrent via `main.py` | Superseded by V14 script |
| `experiment_rtptorrent_v3.yaml` | Industry-style RTPTorrent | Superseded by V14 script |

These exist for reference but the paper results come exclusively from:
1. `experiment_industry_optimized_v3.yaml` (Industry, APFD 0.7611)
2. `experiments/run_filopriori_rtptorrent_v15.py` (RTPTorrent, APFD 0.8540)

---

## Usage

### Industrial Dataset (APFD 0.7611)

```bash
source venv/bin/activate
python main.py --config configs/experiment_industry_optimized_v3.yaml
```

### RTPTorrent (APFD 0.8540)

```bash
source venv/bin/activate
python experiments/run_filopriori_rtptorrent_v15.py
```

---

## Key Configuration Differences Between Datasets

| Aspect | Industrial QTA | RTPTorrent V14 |
|--------|---------------|----------------|
| Model type | `dual_head` | `dual_stream_v8` + DeepOrder DNN |
| Graph | Multi-edge (5 types) | co_failure only |
| Semantic info | Rich (descriptions, steps, commits) | Limited (test names) |
| Learning rate | 3e-5 | 1e-3 |
| Ensemble | No (single model) | Yes (GATv2 + DNN alpha blending) |
| Class balancing | Balanced sampling (29:1) | Focal loss + pos_weight (clamped) |
| Orphan handling | KNN pipeline (k=5) | Not applicable |

See `docs/STABLE_MODEL_PARAMETERS.md` for complete parameter listings.

---

*Maintained by: Filo-Priori Team*
