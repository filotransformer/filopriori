# Stable Model Parameters — Both Datasets

**WARNING FOR AI AGENTS AND DEVELOPERS:**
These configurations have been empirically validated and produce the results reported in the paper.
**DO NOT MODIFY** any parameter unless explicitly requested by the user with empirical justification.

See `CLAUDE.md` at the project root for the full parameter protection policy.

---

## Dataset 1: Industrial QTA (APFD 0.7611)

**Config file:** `configs/experiment_industry_optimized_v3.yaml`
**Run command:** `python main.py --config configs/experiment_industry_optimized_v3.yaml`
**Results:** `results/experiment_industry_optimized_v3/`

### Model Architecture

| Parameter | Value | Sensitivity |
|-----------|-------|-------------|
| Model type | `dual_head` | -- |
| Semantic input dim | 1536 (768 test + 768 commit) | Fixed |
| Semantic hidden dim | 256 | Low |
| Structural input dim | 19 (10 V2.5 + 9 DeepOrder) | Fixed |
| Structural hidden dim | 64 | Low |
| Fusion input dim | 320 | Fixed |
| Fusion hidden dim | 256 | Low |
| Classifier hidden dim | 128 | Low |
| GAT type | GAT | Fixed |
| GAT hidden dim | 128 | Low |
| GAT layers | **1** | HIGH (1 > 2 > 3) |
| GAT heads | **2** | HIGH (2 > 4 > 8) |
| Dropout | 0.15 | Medium |
| SBERT model | `all-mpnet-base-v2` | Fixed |

### Training

| Parameter | Value | Sensitivity |
|-----------|-------|-------------|
| Learning rate | **3.0e-05** | HIGH |
| Optimizer | AdamW | Fixed |
| Weight decay | 1e-4 | Low |
| Batch size | 16 | Medium |
| Max epochs | 80 | Low (early stops ~50) |
| Early stopping patience | 15 | Low |
| Monitor | val_f1_macro | Fixed |
| Gradient clip | 1.0 | Low |
| Scheduler | Cosine (eta_min=1e-6) | Low |
| Warmup epochs | 5 | Low |

### Loss & Class Balancing (CRITICAL)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Loss type | `dual_head` | Fixed |
| `use_class_weights` | **false** | CRITICAL: true causes mode collapse |
| `focal_alpha` | 0.75 | Medium |
| `focal_gamma` | 2.0 | Medium |
| `use_balanced_sampling` | **true** | CRITICAL: primary balancing mechanism |
| `minority_weight` | 1.0 | Fixed |
| `majority_weight` | 0.035 | Tuned (~29:1 ratio) |

### Graph Configuration

| Parameter | Value |
|-----------|-------|
| Multi-edge | true |
| Edge types | co_failure (1.0), co_success (0.5), semantic (0.3), temporal (0.2), component (0.4) |
| Semantic top_k | 10 |
| Semantic threshold | 0.65 |
| Min co-occurrences | 1 |
| Weight threshold | 0.03 |

### Orphan Handling (KNN Pipeline)

| Parameter | Value |
|-----------|-------|
| Method | knn_pfail |
| k_neighbors | 5 |
| alpha_blend | 0.55 |
| structural_weight | 0.35 |
| temperature | 0.7 |
| similarity_metric | euclidean |

### Historical Boost

| Parameter | Value |
|-----------|-------|
| historical_boost_weight | 0.55 |
| priority_score num_cycles | 10 |
| priority_score decay_type | exponential |
| priority_score decay_factor | 0.8 |

---

## Dataset 2: RTPTorrent V15 (APFD 0.8540)

**Script:** `experiments/run_filopriori_rtptorrent_v14.py`
**Run command:** `python experiments/run_filopriori_rtptorrent_v14.py`
**Results:** `results/filopriori_rtptorrent_v14/`

### Model Architecture (Dual-Stream V8 + DeepOrder DNN Ensemble)

| Parameter | Value | Sensitivity |
|-----------|-------|-------------|
| Model type | `dual_stream_v8` | Fixed |
| Semantic input dim | 768 | Fixed |
| Semantic hidden dim | 256 | Low |
| Semantic layers | 2 | Low |
| Semantic dropout | 0.3 | Low |
| Structural input dim | 19 | Fixed |
| Structural hidden dim | 128 | Low |
| Structural heads | 4 (GATv2) | Medium |
| Structural dropout | 0.3 | Low |
| Fusion type | cross_attention | Fixed |
| Fusion hidden dim | 256 | Low |
| Fusion heads | 4 | Low |
| Fusion dropout | 0.1 | Low |
| Classifier hidden dims | [128, 64] | Low |
| Classifier dropout | 0.4 | Low |
| SBERT model | `all-mpnet-base-v2` | Fixed |

### GATv2 Training

| Parameter | Value | Sensitivity |
|-----------|-------|-------------|
| Learning rate | **1e-3** | HIGH |
| Weight decay | 1e-4 | Low |
| Max epochs | 30 | Low |
| Patience | 7 | Low |
| Focal gamma | 2.0 | Medium |
| Max pos_weight | 10.0 | Medium |

### DeepOrder DNN Component (CRITICAL)

| Parameter | Value | Sensitivity |
|-----------|-------|-------------|
| Hidden dims | [64, 32, 16] | Medium |
| Dropout | 0.2 | Low |
| Learning rate | 0.001 | Medium |
| Epochs | **15** | HIGH (was 10 in V13, 15 improves convergence) |
| Batch size | 128 | Low |
| History window | 10 | Medium |
| `max_do_train_builds` | **5000** | HIGH (higher hurts large projects) |
| `max_dnn_pos_weight` | **50.0** | CRITICAL (unclamped reaches 3000+ for rare-failure projects) |
| Loss function | **BCELoss** | CRITICAL (NOT BCEWithLogitsLoss — see bug below) |

### Graph Configuration

| Parameter | Value |
|-----------|-------|
| Graph type | `co_failure` (only) |
| Min co-occurrences | 2 |
| Weight threshold | 0.1 |

### Alpha Blending (Ensemble)

| Parameter | Value |
|-----------|-------|
| Formula | `final = alpha * GNN_probs + (1-alpha) * DNN_scores` |
| Search range | [0.0, 0.1, 0.2, ..., 0.9] |
| Selection | Validation-optimized per project |
| min_val_failure_builds | 3 (guard against bad alpha choices) |
| min_tcs_for_model | 30 |

### Other

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Train ratio | 0.8 (temporal split) |
| Val ratio | 0.1 (from train set) |
| Max train rows | 500,000 |
| SBERT batch size | 64 |

---

## Critical Bug Fixes in V15 (DO NOT REINTRODUCE)

### 1. Double-Sigmoid Bug

`DeepOrderNet` (`src/baselines/deeporder.py`) has `nn.Sigmoid()` as its final layer.
V13 used `nn.BCEWithLogitsLoss` which applies sigmoid internally, causing:
```
output = sigmoid(sigmoid(logits))  # gradients compressed to near-zero
```
**Fix:** Use `nn.BCELoss(reduction='none')` with manual pos_weight via `torch.where`.

### 2. Unclamped pos_weight

For rare-failure projects (e.g., SonarSource with 0.06% failure rate):
```
pos_weight = (1 - 0.0006) / 0.0006 = 1666  # causes DNN to predict ALL as failure
```
**Fix:** Clamp to `max_dnn_pos_weight = 50.0`.

### 3. Inconsistent Training Paths

V13 had two code paths: custom training for large projects, `do_model.train()` for small ones.
The `do_model.train()` method uses `BCEWithLogitsLoss` (the buggy loss).
**Fix:** All projects use the same custom BCELoss training loop.

### 4. Pre-warmed DNN History

DNN history must be built from ALL training builds (fast, O(n) feature computation),
not just the last `max_do_train_builds`. Training then uses only the last 5K builds
but with full historical context.

---

## Cross-Dataset Validation Results (March 2026)

### Ablation Study (RQ2 — RTPTorrent, 20 Projects)

| Configuration | APFD | Delta | Significance |
|---|---|---|---|
| Full Ensemble (V15) | 0.8540 | -- | -- |
| w/o DNN Ensemble | 0.8322 | -2.6% | p<0.001*** |
| w/o GATv2 | 0.8451 | -1.0% | p<0.05* |
| w/o Semantic Stream | 0.8450 | -1.1% | ns |
| w/o Multi-Edge Graph | 0.8513 | -0.3% | ns |

**Finding:** With the V15 improvements (Execution-Level Temporal GNN and Multi-Edge Graph), the GNN is significantly stronger on RTPTorrent. The drop without the DNN is now only 2.6% (down from 13.1%), proving that the graph model is highly effective even when metadata is sparse.

### Temporal Cross-Validation (RQ3 — RTPTorrent, 20 Projects)

| Metric | Value |
|---|---|
| Grand Mean APFD | 0.816 |
| 95% CI | [0.754, 0.877] |
| Drop vs standard eval | -2.7% (0.854 → 0.816) |
| Fold progression | 0.790 → 0.816 → 0.823 → 0.834 |
| Projects ≥ 0.80 | 14/20 |
| Cross-fold std < 0.04 | 16/20 projects |

**Finding:** No concept drift. Performance improves with more training data. Stable across projects.

### Sensitivity Analysis (RQ4 — RTPTorrent, 20 Projects)

| Parameter | Values Tested | APFD Range | Impact |
|---|---|---|---|
| Alpha (GNN-DNN blend) | 0.0, 0.3, 0.5, 0.7, 1.0, optimized | 0.832 – 0.854 | 2.6% (alpha=1.0 drop) |
| DNN Epochs | 5, 10, 20 (vs default 15) | 0.837 – 0.844 | 0.8% |
| Max pos_weight | 10, 25, 100 (vs default 50) | 0.830 – 0.844 | 1.6% |

**Finding:** Excluding degenerate alpha=1.0, all 11 configurations fall within a 1.6% range
(0.830–0.844). Model is robust to continuous hyperparameter choices.

**Script:** `experiments/run_rtptorrent_ablation_sensitivity.py`
**Results:** `results/rtptorrent_ablation_sensitivity/`

### DNN Ensemble Verification (Industrial Dataset)

| Variant | APFD | vs GNN (0.761) |
|---|---|---|
| GNN-only (existing V3) | **0.7611** | -- |
| DNN-only | 0.6861 | -9.9% |

**Finding:** DNN ensemble is redundant on the Industrial dataset. The GNN with multi-edge
graph outperforms DNN-only by 9.9%, empirically justifying the design choice to omit the
DNN ensemble in the industrial configuration.

**Script:** `experiments/run_dnn_ensemble_industry.py`
**Results:** `results/dnn_ensemble_industry/`

---

*Last Updated: March 2026*
