# Filo-Priori: A Multi-Edge Graph Attention Approach to Test Case Prioritization

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyG](https://img.shields.io/badge/PyG-2.3+-orange.svg)
![APFD](https://img.shields.io/badge/APFD-0.761-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A **graph-based deep learning framework** for intelligent test case prioritization in CI/CD pipelines. Centers on GATv2 graph attention over configurable test relationship graphs, complemented by a DNN ensemble for semantically-sparse settings.

**Target Journal:** IEEE Transactions on Software Engineering (IEEE TSE)

**Authors:** Acauan C. Ribeiro, Eduardo L. Feitosa, Andre L. da Costa Carvalho, Eulanda M. dos Santos, Bruno F. Gadelha, Yan R. Soares, and Jose Nascimento

**Affiliation:** Instituto de Computacao (IComp) - Universidade Federal do Amazonas (UFAM) / Motorola Mobility LLC

---

## Abstract

Test Case Prioritization (TCP) aims to order test cases to maximize early fault detection in Continuous Integration (CI) environments. Existing approaches treat test cases as independent entities, ignoring the structural relationships that characterize real-world testing.

**Filo-Priori** is a graph-based deep learning framework that explicitly models test case relationships through two evaluated variants:

1. **Filo-Priori-Full** (rich metadata): GATv2 over a multi-edge graph (up to 5 edge types) with KNN orphan node imputation
2. **Filo-Priori-Ensemble** (sparse metadata): GATv2 over co-failure graph + DeepOrder-inspired DNN with validation-optimized α-blending
3. **Key negative result**: Generic semantic embeddings (Sentence-BERT) provide no statistically significant improvement (p=0.309), indicating that graph-based structural modeling subsumes the need for textual features in TCP
4. **Simplified Dual-Balancing**: Balanced sampling (29:1) + Focal Loss (α_f=0.75, γ=2.0) for severe class imbalance

---

## Key Results

### Summary Across Two Datasets

| Experiment | APFD | vs. Best Baseline |
|---|---|---|
| Industrial QTA | **0.761** | +10.2% (DeepOrder) |
| RTPTorrent (20 projects) | **0.854** | +1.6% (TCP-Net) |
| **Overall Average** | **0.799** | -- |

### Industrial Dataset - 277 Builds with Failures

| Method | APFD | Std | vs Filo-Priori |
|---|---|---|---|
| **Filo-Priori** | **0.7611** | **0.189** | -- |
| DeepOrder | 0.6890 | 0.266 | +10.2% |
| TCP-Net | 0.6704 | 0.271 | +13.3% |
| NodeRank | 0.6609 | 0.270 | +14.9% |
| RETECS | 0.6406 | 0.281 | +18.6% |
| FailRank-BB | 0.5953 | 0.263 | +27.6% |

### RTPTorrent - 20 Open-Source Java Projects

#### Deep Learning Baselines

| Method | APFD | Std | vs Filo-Priori |
|---|---|---|---|
| **Filo-Priori** | **0.8540** | 0.112 | -- |
| TCP-Net | 0.8253 | 0.110 | +1.6% |
| FailRank-BB | 0.8218 | 0.092 | +2.0% |
| DeepOrder | 0.8136 | 0.104 | +3.0% |
| NodeRank | 0.8038 | 0.109 | +4.3% |
| RETECS | 0.6791 | 0.156 | +23.5% |

#### Heuristic Baselines

| Method | APFD | vs Filo-Priori |
|---|---|---|
| **Filo-Priori** | **0.8540** | -- |
| recently_failed | 0.8209 | +2.1% |
| optimal_duration | 0.5934 | +41.3% |
| matrix_naive | 0.5693 | +47.3% |
| random | 0.4940 | +69.7% |
| optimal_failure (oracle) | 0.9249 | -9.3% |

---

## Research Questions

| RQ | Question | Answer |
|---|---|---|
| **RQ1** | Can modeling structural relationships between test cases through graph neural networks improve fault detection compared to approaches that treat tests independently? | **Industrial:** APFD 0.761, +10.2% to +27.6% over all baselines (p<0.001). **RTPTorrent:** APFD 0.854, highest among all methods; robust improvements over NodeRank (+4.3%) and RETECS (+23.5%), narrow margins over TCP-Net (+1.6%), FailRank-BB (+2.0%), DeepOrder (+3.0%) |
| **RQ2** | Which architectural components contribute most to the effectiveness of graph-based test case prioritization? | **Industrial:** Graph (+10.0%), Orphan KNN (+5.9%), Balancing (+4.0%). **RTPTorrent:** DNN ensemble (+13.1%), all others non-significant. Cross-dataset: architecture adapts to metadata richness |
| **RQ3** | How robust is the proposed approach when applied to builds from different time periods than those used for training? | **Industrial:** Temporal CV APFD 0.663 (-12.9% vs standard). **RTPTorrent:** Temporal CV APFD 0.816 (-2.7% vs standard). Both stable across folds, no concept drift |
| **RQ4** | How sensitive is the approach to the choice of key hyperparameters? | **Industrial:** Max impact 3.6% (loss function). **RTPTorrent:** Max impact 1.6% (excl. degenerate alpha=1.0). Both datasets confirm robustness to hyperparameter choices |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run best configuration (Industrial Dataset)
python main.py --config configs/experiment_industry_optimized_v3.yaml

# Run RTPTorrent evaluation (V15 — produces APFD 0.8540)
python experiments/run_filopriori_rtptorrent_v15.py

# Run RTPTorrent ablation + sensitivity + temporal CV
python experiments/run_rtptorrent_ablation_sensitivity.py --all

# Run DNN ensemble verification on Industrial dataset
python experiments/run_dnn_ensemble_industry.py
```

---

## Architecture Overview

```
+------------------------------------------------------------------------+
|               FILO-PRIORI CONFIGURABLE FRAMEWORK                       |
+------------------------------------------------------------------------+
|                                                                        |
|  +-------------------+                     +-------------------+       |
|  |  SEMANTIC INPUT   |                     | STRUCTURAL INPUT  |       |
|  |  (depends on      |                     |  19 Features      |       |
|  |   metadata:       |                     |  (10 base +       |       |
|  |   768 or 1536-dim)|                     |   9 DeepOrder)    |       |
|  +--------+----------+                     +--------+----------+       |
|           |                                         |                  |
|           v                                         v                  |
|  +-------------------+                     +-------------------+       |
|  |  SBERT Encoder    |                     |  Test Relationship|       |
|  |  all-mpnet-base-v2|                     |  Graph (1-5 edge  |       |
|  |                   |                     |  types by metadata)|      |
|  +--------+----------+                     +--------+----------+       |
|           |                                         |                  |
|           v                                         v                  |
|  +-------------------+                     +-------------------+       |
|  | SEMANTIC STREAM   |                     |STRUCTURAL STREAM  |       |
|  | FFN + residual    |                     | GATv2 attention   |       |
|  | (complementary)   |                     | (primary signal)  |       |
|  +--------+----------+                     +--------+----------+       |
|           |                                         |                  |
|           +----------------+--------------------+                      |
|                            v                                           |
|                   +-------------------+                                |
|                   | CROSS-ATTENTION   |                                |
|                   | FUSION (512-dim)  |                                |
|                   +--------+----------+                                |
|                            |                                           |
|                            v                                           |
|                   +-------------------+                                |
|                   |   CLASSIFIER      |                                |
|                   |  MLP [128->64->2] |                                |
|                   +--------+----------+                                |
|                            |                                           |
|           +----------------+----------------+                          |
|           v                                 v                          |
|  +-------------------+            +-------------------+                |
|  | ORPHAN HANDLING   |            |  DNN ENSEMBLE     |                |
|  | KNN pipeline      |            |  DeepOrder DNN    |                |
|  | (rich metadata)   |            |  + alpha blending |                |
|  +--------+----------+            |  (sparse metadata)|                |
|           |                       +--------+----------+                |
|           v                                v                           |
|        P(Fail) -----> alpha-optimized blending -----> Ranked List      |
|                                                                        |
+------------------------------------------------------------------------+
```

---

## Key Innovations

### 1. Configurable Test Relationship Graph

Co-failure edges form the universal base (available from any CI logs). When richer metadata exists, the graph extends to 5 edge types:

| Edge Type | Weight | Available When |
|---|---|---|
| Co-Failure | 1.0 | Always (CI logs) |
| Co-Success | 0.5 | Always (CI logs) |
| Component | 0.4 | Component labels exist |
| Semantic | 0.3 | Rich test descriptions |
| Temporal | 0.2 | Execution order data |

With all types: density 0.02% → 0.5-1.0%, **77.4%** of tests connected.

### 2. GATv2 Structural Stream + DNN Ensemble

- **Structural Stream** (primary): GATv2 (1 layer, 2 heads) over 19 features on the test graph → 256-dim
- **DNN Ensemble** (Filo-Priori-Ensemble): DeepOrder-inspired DNN with validation-optimized α-blending
- **Semantic Stream** (investigated): SBERT embeddings + cross-attention fusion — ablation shows no significant benefit (p=0.309)

### 3. Simplified Dual-Balancing

Avoids mode collapse by separating prior correction (sampling) from gradient focusing (focal loss):

| Mechanism | Triple-Comp. | Dual-Balancing |
|---|---|---|
| Class weights in loss | 19x | **Disabled** |
| Focal α_f | 0.85 (1.7x) | 0.75 (mild) |
| Balanced sampling | 20x | 29x (only) |
| Effective weight | ~646x | **29x** |

### 4. Deployment-Specific Extensions

**Orphan Handling** (rich metadata): 4-stage KNN pipeline (k=5, cosine similarity, T=0.7, alpha=0.55). Restores orphan score variance from 0.0 to 0.046.

**DNN Ensemble** (sparse metadata): DeepOrder DNN + alpha blending. Alpha auto-optimized per project on validation set. DNN-only achieves APFD 0.686 on industrial (-9.9% vs GNN), confirming GNN superiority when metadata is rich.

---

## Ablation Study (RQ2)

### Industrial Dataset

| Configuration | APFD | Delta | p-value |
|---|---|---|---|
| Full Model | 0.7611 | -- | -- |
| w/o Enriched Multi-Edge Graph | 0.6835 | -10.0% | <0.001*** |
| w/o Orphan KNN Scoring | 0.7145 | -5.9% | <0.001*** |
| w/o Single Balancing | 0.7291 | -4.0% | <0.001*** |
| w/o DeepOrder Features | 0.7443 | -2.0% | 0.003** |
| w/o Threshold Optimization | 0.7519 | -1.0% | 0.042* |

### RTPTorrent (20 Projects)

| Configuration | APFD | Delta | Sig. |
|---|---|---|---|
| Full Model | 0.8540 | -- | -- |
| w/o DNN Ensemble | 0.8322 | -2.6% | p<0.001*** |
| w/o GATv2 | 0.8451 | -1.0% | p<0.05* |
| w/o Semantic Stream | 0.8450 | -1.1% | ns |
| w/o Multi-Edge Graph | 0.8513 | -0.3% | ns |

**Cross-dataset insight:** On Industrial data, the graph and GATv2 are dominant (+10%, +17%). On RTPTorrent, the DNN ensemble is the primary driver, though the improved GNN substantially reduced reliance on it (drop without DNN went from -13.1% to -2.6%). **Negative result**: Semantic stream provides no significant improvement on either dataset.

---

## Temporal Validation (RQ3)

### Industrial Dataset

| Validation Method | APFD | Std | N |
|---|---|---|---|
| Temporal 5-Fold CV | 0.6629 | 0.279 | 215 |
| Sliding Window CV | 0.6279 | 0.272 | 248 |
| Concept Drift Test | 0.6187 | 0.277 | 152 |

### RTPTorrent (20 Projects, 4-Fold Temporal CV)

| Metric | Value |
|---|---|
| Grand Mean APFD | **0.816** |
| 95% CI | [0.754, 0.877] |
| Drop from standard eval (0.854) | -2.7% |
| Fold progression | 0.790 → 0.816 → 0.823 → 0.834 |
| Projects with APFD ≥ 0.80 | 14/20 |

**Cross-dataset insight:** Both datasets show graceful degradation under temporal constraints with no concept drift. RTPTorrent's smaller drop (-2.7% vs -12.9%) reflects larger training sets in multi-project evaluation.

---

## Sensitivity Analysis (RQ4)

### Industrial Dataset

| Parameter | Values Tested | Best | Impact |
|---|---|---|---|
| Loss Function | CE, Focal, W. Focal | W. CE | 3.6% |
| Learning Rate | 3e-5, 5e-5 | 3e-5 | 2.7% |
| GNN Architecture | 1L/2H, 2L/4H | 1L/2H | 2.7% |
| Structural Features | 6, 10, 29 | 10 | 1.7% |

### RTPTorrent (20 Projects)

| Parameter | Values Tested | Best | Impact |
|---|---|---|---|
| Alpha (GNN-DNN blend) | 0.0, 0.3, 0.5, 0.7, 1.0 | 0.0-0.7 (stable) | 13.1% (alpha=1.0 critical) |
| DNN Epochs | 5, 10, 15, 20 | 15-20 | 0.8% |
| Max pos_weight | 10, 25, 50, 100 | 50-100 | 1.6% |

**Cross-dataset insight:** Both datasets confirm robustness to continuous hyperparameters. Industrial max range: 3.6%. RTPTorrent max range: 1.6% (excl. degenerate alpha=1.0). The most impactful decision is architectural (including the DNN ensemble), not hyperparameter tuning.

---

## Structural Features (19 total)

### Base Features (10)

| Feature | Description |
|---|---|
| test_age | Builds since first appearance |
| failure_rate | Historical failure ratio |
| recent_failure_rate | Failure rate in last 5 builds |
| flakiness_rate | Pass-Fail oscillation frequency |
| consecutive_failures | Current failure streak length |
| max_consecutive_failures | Worst-case failure streak |
| failure_trend | Recent - overall failure rate |
| commit_count | Unique associated commits |
| cr_count | Distinct change requests |
| test_novelty | Binary (1 if first appearance) |

### DeepOrder-Inspired Features (9)

| Feature | Description |
|---|---|
| execution_status_last_1 | Last execution result |
| execution_status_last_2 | Failure proportion in last 2 |
| execution_status_last_3 | Failure proportion in last 3 |
| execution_status_last_5 | Failure proportion in last 5 |
| execution_status_last_10 | Failure proportion in last 10 |
| cycles_since_last_fail | Normalized builds since last failure |
| distance | Temporal distance from last failure |
| status_changes | Total Pass-Fail transitions |
| fail_rate_last_10 | Failure rate over last 10 executions |

---

## Datasets

### Dataset 1: Industrial QTA

| Statistic | Value |
|---|---|
| Total Executions | 52,102 |
| Unique Builds | 1,339 |
| Builds with Failures | 277 (20.7%) |
| Unique Test Cases | 2,347 |
| Pass:Fail Ratio | 37:1 |
| Semantic Info | Rich (descriptions, steps, commits, diffs) |
| Structural Info | Execution history, failure patterns |

### Dataset 2: RTPTorrent

| Statistic | Value |
|---|---|
| Projects | 20 Java projects |
| Total Builds | >100,000 |
| Source | Travis CI build logs |
| Semantic Info | Limited (test names only) |
| License | CC BY 4.0 |

---

## Project Structure

```
filo-priori/
├── CLAUDE.md                        # AI agent instructions + frozen parameters
├── main.py                          # Main entry point (Industrial dataset)
├── requirements.txt                 # Dependencies
├── configs/
│   └── experiment_industry_optimized_v3.yaml  # Industry config (APFD 0.7611) - FROZEN
├── experiments/
│   ├── run_filopriori_rtptorrent_v15.py       # Filo-Priori V15 (APFD 0.8540) - FROZEN
│   ├── run_rtptorrent_ablation_sensitivity.py # Ablation + Sensitivity + Temporal CV (RTPTorrent)
│   ├── run_dnn_ensemble_industry.py           # DNN ensemble verification on Industrial
│   ├── run_deeporder_*.py                     # DeepOrder baselines
│   ├── run_noderank_*.py                      # NodeRank baselines
│   ├── run_retecs_*.py                        # RETECS baselines
│   ├── run_tcpnet_*.py                        # TCP-Net baselines
│   ├── run_failrank_bb_*.py                   # FailRank-BB baselines
│   └── archived/                              # Old Filo-Priori versions (not in paper)
├── paper/
│   ├── main_ieee_tse.tex            # IEEE TSE paper
│   ├── references_ieee.bib          # Bibliography
│   ├── figures/                     # Paper figures (PDF + LaTeX source)
│   ├── sections/                    # Paper sections (results, discussion, threats)
│   └── tables/                      # LaTeX tables
├── src/
│   ├── models/
│   │   ├── dual_stream_v8.py        # Main dual-stream model
│   │   ├── cross_attention.py       # Cross-attention fusion module
│   │   └── model_factory.py         # Model creation utility
│   ├── layers/
│   │   └── gatv2.py                 # Graph Attention Networks v2
│   ├── embeddings/
│   │   └── sbert_encoder.py         # Sentence-BERT encoder
│   ├── evaluation/
│   │   ├── apfd.py                  # APFD metric
│   │   └── orphan_ranker.py         # KNN-based orphan scoring
│   ├── phylogenetic/
│   │   └── multi_edge_graph_builder.py  # Multi-edge graph construction
│   ├── preprocessing/
│   │   └── structural_feature_extractor_v2_5.py  # 19 structural features
│   ├── baselines/
│   │   ├── deeporder.py             # DeepOrder baseline
│   │   ├── noderank.py              # NodeRank baseline
│   │   ├── retecs.py                # RETECS baseline
│   │   └── failrank_bb.py           # FailRank-BB baseline
│   └── training/
│       └── losses.py                # Loss functions (Focal Loss)
├── datasets/
│   ├── 01_industry/                 # Industrial QTA dataset
│   └── 02_rtptorrent/               # RTPTorrent open-source dataset
├── results/
│   ├── experiment_industry_optimized_v3/   # Industry results (APFD 0.7611)
│   ├── filopriori_rtptorrent_v15/          # RTPTorrent V15 results (APFD 0.8540)
│   ├── rtptorrent_ablation_sensitivity/    # RTPTorrent ablation + sensitivity + temporal CV
│   ├── dnn_ensemble_industry/              # DNN ensemble verification (APFD 0.686)
│   ├── deeporder_*/                        # DeepOrder baseline results
│   ├── noderank_*/                         # NodeRank baseline results
│   ├── retecs_*/                           # RETECS baseline results
│   ├── tcpnet_*/                           # TCP-Net baseline results
│   └── failrank_bb_*/                      # FailRank-BB baseline results
├── docs/
│   ├── BASELINE_RESULTS.md          # Baseline comparison (all datasets)
│   ├── STABLE_MODEL_PARAMETERS.md   # Frozen hyperparameters reference
│   ├── TECHNICAL_REPORT_APFD_0.7611.md
│   └── PIPELINE_ARCHITECTURE.md
└── cache/                           # Pre-computed embeddings and graphs
```

---

## Training Configuration (Industrial — Filo-Priori-Full)

| Parameter | Value |
|---|---|
| Framework | PyTorch 2.0, PyTorch Geometric 2.3 |
| Hardware | NVIDIA RTX 3090 (24GB VRAM) |
| Optimizer | AdamW (weight_decay=1e-4) |
| Learning Rate | 3e-5 with cosine annealing |
| Batch Size | 16 |
| Epochs | 80 (early stop patience=15) |
| Loss | Focal Loss (α_f=0.75, γ=2.0) |
| Balanced Sampling | 29:1 (minority:majority) |
| Gradient Clipping | max norm 1.0 |
| Monitoring | val_f1_macro |

---

## Baselines

### Heuristic Baselines

- **Random**: Random ordering (APFD ~ 0.5)
- **Recency**: Prioritizes recently failed tests
- **RecentFailureRate**: Failure rate in last 5 builds
- **FailureRate**: Overall historical failure rate
- **GreedyHistorical**: Multi-heuristic greedy selection

### ML Baselines

- **Logistic Regression**: Linear classifier on structural features
- **Random Forest**: Ensemble decision trees
- **XGBoost**: Gradient boosting

### Deep Learning Baselines

- **DeepOrder**: DNN with 8 historical features
- **NodeRank**: Mutation-based analysis with ensemble learning
- **RETECS**: Reinforcement learning for TCP in CI
- **TCP-Net**: End-to-end DNN with temporal execution features
- **FailRank-BB**: BERT embeddings + LogisticRegression (Hernandes et al., 2024)

---

## Documentation

| Document | Description |
|---|---|
| [CLAUDE.md](CLAUDE.md) | AI agent instructions, frozen parameters, valid experiments |
| [Paper (LaTeX)](paper/main_ieee_tse.tex) | Full IEEE TSE paper |
| [BASELINE_RESULTS.md](docs/BASELINE_RESULTS.md) | Baseline comparison (all datasets) |
| [STABLE_MODEL_PARAMETERS.md](docs/STABLE_MODEL_PARAMETERS.md) | Frozen hyperparameters reference |
| [TECHNICAL_REPORT_APFD_0.7611.md](docs/TECHNICAL_REPORT_APFD_0.7611.md) | Detailed APFD analysis |
| [PIPELINE_ARCHITECTURE.md](docs/PIPELINE_ARCHITECTURE.md) | Visual diagrams (Mermaid) |

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 16GB | 32GB |
| GPU VRAM | 8GB | 12GB+ |
| CUDA | 11.8+ | 12.1+ |

### Software

```
torch>=2.0.0
torch-geometric>=2.3.0
sentence-transformers>=2.2.2
transformers>=4.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.10.0
PyYAML>=6.0
```

---

## Citation

```bibtex
@article{ribeiro2026filopriori,
  title={Filo-Priori: A Multi-Edge Graph Attention Approach to
         Test Case Prioritization},
  author={Ribeiro, Acauan C. and Feitosa, Eduardo L. and
          Carvalho, Andre L. da Costa and Santos, Eulanda M. dos and
          Gadelha, Bruno F. and Soares, Yan R. and Nascimento, Jose},
  journal={IEEE Transactions on Software Engineering},
  year={2026},
  note={Under Review}
}
```

---

## References

- **GAT**: Velickovic, P., et al. (2018). Graph Attention Networks. ICLR.
- **GATv2**: Brody, S., et al. (2022). How Attentive are Graph Attention Networks? ICLR.
- **Focal Loss**: Lin, T., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
- **SBERT**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT. EMNLP.
- **DeepOrder**: Chen, J., et al. (2023). Deep Learning for TCP in CI Testing. TSE.
- **NodeRank**: Li, Z., et al. (2024). NodeRank: Test Input Prioritization for GNNs.
- **RTPTorrent**: Mattis, T., et al. (2020). RTPTorrent: An Open-Source Dataset. MSR.
- **Hernandes et al.**: Hernandes, V., et al. (2025). A Method for Regression Testing Plan Ordering.

---

## License

MIT License - see [LICENSE](LICENSE) for details.
