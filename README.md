# Filo-Priori: Co-Failure Graph Attention for Test Case Prioritization in Continuous Integration

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyG](https://img.shields.io/badge/PyG-2.3+-orange.svg)
![APFD-Industrial](https://img.shields.io/badge/APFD%20Industrial-0.761-brightgreen.svg)
![APFD-RTPTorrent](https://img.shields.io/badge/APFD%20RTPTorrent-0.854-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Submitted%20to%20IEEE%20TSE-blue.svg)

A **graph-based deep learning framework** for intelligent test case prioritization in CI/CD pipelines. Centers on **GATv2 graph attention over a co-failure test relationship graph**, complemented by a DNN ensemble with data-driven blending for semantically-sparse settings.

**Target Journal:** IEEE Transactions on Software Engineering (IEEE TSE)
**Submission Type:** Regular (Journal First)
**Replication Package:** https://github.com/filotransformer/filopriori

**Authors:** Acauan C. Ribeiro, Eduardo L. Feitosa, Andre L. da Costa Carvalho, Eulanda M. dos Santos, Bruno F. Gadelha, Yan R. Soares, and Jose Nascimento

**Affiliations:**
- Instituto de Computação (IComp), Universidade Federal do Amazonas (UFAM), Manaus, AM, Brazil
- Motorola Mobility Comércio de Produtos Eletrônicos Ltda., Manaus, AM, Brazil

---

## Abstract

In Continuous Integration (CI), growing test suites make exhaustive testing impractical, motivating Test Case Prioritization (TCP) to maximize early fault detection. Existing TCP approaches typically treat tests as independent entities, ignoring structural relationships: tests that co-fail share underlying dependencies, and execution patterns reveal systematic connections.

**Filo-Priori** is a graph-based deep learning framework that models inter-test relationships through Graph Attention Networks (GATv2) over a co-failure graph, complemented by a Deep Neural Network (DNN) ensemble with data-driven blending for semantically-sparse settings. The framework provides two evaluated configurations:

1. **Filo-Priori-Full** (metadata-rich setting): GATv2 over a multi-edge graph (up to 5 edge types) with KNN orphan node imputation. GNN probability is the final score.
2. **Filo-Priori-Ensemble** (metadata-sparse setting): GATv2 over a co-failure graph + DeepOrder-inspired DNN with validation-optimized α-blending.

**Key Findings:**
- **Co-failure edges are the decisive graph component** (+17.0% in isolation, p < 0.001); supplementary edge types do not improve aggregate performance.
- **Negative result on semantic features:** Generic Sentence-BERT embeddings provide no statistically significant improvement (p = 0.309), confirmed by a Random-Fixed control experiment (p = 0.965).
- **Simplified Dual-Balancing**: Balanced sampling (29:1) + Focal Loss (α_f=0.75, γ=2.0) prevents mode collapse observed under triple-compensation.

---

## Key Results

### Summary Across Two Datasets

| Experiment | Variant | APFD | vs. Best Baseline |
|---|---|---|---|
| Industrial QTA | Filo-Priori-Full | **0.761** | +10.2% (DeepOrder, p<0.001) |
| RTPTorrent (20 projects) | Filo-Priori-Ensemble | **0.854** | +1.6% (TCP-Net, ns after correction) |
| **Overall Average** | -- | **0.800** | -- |

### Industrial Dataset — 277 Builds with Failures

| Method | APFD | Std | p-value | Cliff's δ | Δ |
|---|---|---|---|---|---|
| **Filo-Priori-Full** | **0.761** | **0.189** | -- | -- | -- |
| DeepOrder | 0.689 | 0.266 | <0.001 | 0.10 (N) | +10.2% |
| TCP-Net | 0.670 | 0.271 | <0.001 | 0.14 (N) | +13.3% |
| NodeRank | 0.661 | 0.270 | <0.001 | 0.14 (N) | +14.9% |
| RETECS | 0.641 | 0.281 | <0.001 | 0.21 (S) | +18.6% |
| FailRank-BB | 0.595 | 0.263 | <0.001 | 0.35 (M) | +27.6% |

All industrial comparisons significant at p<0.001 and survive Bonferroni correction.

### RTPTorrent — 20 Open-Source Java Projects

| Method | APFD | p-value | Holm-Bonferroni | Δ |
|---|---|---|---|---|
| **Filo-Priori-Ensemble** | **0.854** | -- | -- | -- |
| TCP-Net | 0.825 | 0.087 | ns | +1.6% |
| FailRank-BB | 0.822 | 0.046 | ns (adj ≈0.18) | +2.0% |
| DeepOrder | 0.814 | 0.018 | marginal (adj ≈0.054) | +3.0% |
| NodeRank | 0.809 | 0.009 | **sig** | +5.6% |
| RETECS | 0.679 | <0.001 | **sig** | +23.5% |

After Holm-Bonferroni correction, only NodeRank and RETECS remain significant. Narrow margins over TCP-Net, FailRank-BB, and DeepOrder represent a statistical tie.

---

## Research Questions

| RQ | Question | Answer |
|---|---|---|
| **RQ1** | Can modeling structural relationships between test cases through graph neural networks improve fault detection compared to approaches that treat tests independently? | **Industrial:** APFD 0.761, +10.2% to +27.6% over all baselines (p<0.001). **RTPTorrent:** APFD 0.854, highest among all methods; robust improvements over NodeRank (+5.6%) and RETECS (+23.5%) after correction |
| **RQ2** | Which architectural components contribute most to the effectiveness of graph-based test case prioritization? | **Industrial:** GATv2 graph (+17.0%, p<0.001), Orphan KNN (+5.9%), Balancing (+4.0%). **RTPTorrent:** DNN ensemble (-2.6% without it, p<0.001). Semantic stream non-significant on both datasets |
| **RQ3** | How robust is the proposed approach when applied to builds from different time periods than those used for training? | **Industrial:** Temporal CV APFD 0.619-0.663 (13-19% degradation, expected). **RTPTorrent:** Grand Mean APFD 0.816 (-2.7% vs standard). No concept drift |
| **RQ4** | How sensitive is the approach to the choice of key hyperparameters? | **Industrial:** Max impact 3.6%. **RTPTorrent:** Max impact 1.6% (excl. degenerate alpha=1.0). Both datasets confirm robustness to continuous hyperparameters |

---

## Quick Start

```bash
# Activate venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run best configuration (Industrial Dataset) — APFD 0.7611
python main.py --config configs/experiment_industry_optimized_v3.yaml

# Run RTPTorrent evaluation (V14, produces APFD 0.8540)
python experiments/run_filopriori_rtptorrent_v14.py

# Run RTPTorrent ablation + sensitivity + temporal CV (~14h)
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
|  |  (optional)       |                     |  19 Features      |       |
|  |  768 or 1536-dim  |                     |  (10 base +       |       |
|  |                   |                     |   9 DeepOrder)    |       |
|  +--------+----------+                     +--------+----------+       |
|           |                                         |                  |
|           v                                         v                  |
|  +-------------------+                     +-------------------+       |
|  |  SBERT Encoder    |                     |  Co-Failure Graph |       |
|  |  all-mpnet-base-v2|                     |  (CORE — always   |       |
|  |  (no significant  |                     |   available)      |       |
|  |   benefit, p=0.309)|                    |  + 4 optional     |       |
|  |                   |                     |  edge types       |       |
|  +--------+----------+                     +--------+----------+       |
|           |                                         |                  |
|           v                                         v                  |
|  +-------------------+                     +-------------------+       |
|  | SEMANTIC STREAM   |                     |STRUCTURAL STREAM  |       |
|  | FFN + residual    |                     | GATv2 attention   |       |
|  | (exploratory)     |                     | (PRIMARY signal)  |       |
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
|  | (Filo-Priori-Full)|            |  + α-blending     |                |
|  +--------+----------+            | (Filo-Priori-     |                |
|           |                       |  Ensemble)        |                |
|           |                       +--------+----------+                |
|           v                                v                           |
|        P(Fail) ----> α-optimized blending ----> Ranked List            |
|                                                                        |
+------------------------------------------------------------------------+
```

---

## Key Innovations

### 1. Co-Failure Test Relationship Graph (Core)

Co-failure edges are universally available from CI logs and provide the decisive predictive signal. When richer metadata exists, the graph extends to 5 edge types:

| Edge Type | Weight | Available When | Per-edge-type ablation impact |
|---|---|---|---|
| **Co-Failure** | 1.0 | Always (CI logs) | **+17.0%** in isolation (p<0.001) |
| Co-Success | 0.5 | Always (CI logs) | -0.7% (ns) when added |
| Component | 0.4 | Component labels exist | ns when added |
| Semantic | 0.3 | Rich test descriptions | ns when added |
| Temporal | 0.2 | Execution order data | ns when added |

With all edge types: density 0.02% → 0.5-1.0%, **77.4%** of tests connected (vs 50-60% with co-failure alone). However, the per-edge-type ablation confirms co-failure edges alone are sufficient for optimal aggregate performance.

### 2. GATv2 Structural Stream + DNN Ensemble

- **Structural Stream** (primary): GATv2 (1 layer, 2 heads) over 19 features on the co-failure graph → 256-dim
- **DNN Ensemble** (Filo-Priori-Ensemble): DeepOrder-inspired DNN with validation-optimized α-blending
- **Semantic Stream** (exploratory): SBERT embeddings + cross-attention fusion — ablation shows no significant benefit (p=0.309 industrial; ns on RTPTorrent)

### 3. Simplified Dual-Balancing

Avoids mode collapse by separating prior correction (sampling) from gradient focusing (focal loss):

| Mechanism | Triple-Comp. (broken) | Dual-Balancing (used) |
|---|---|---|
| Class weights in loss | 19x | **Disabled** |
| Focal α_f | 0.85 (1.7x) | 0.75 (mild) |
| Balanced sampling | 20x | 29x (only) |
| Effective weight | ~646x → mode collapse | **29x** → stable |

### 4. Deployment-Specific Extensions

**Orphan Handling** (Filo-Priori-Full): 4-stage KNN pipeline (k=5, cosine similarity, T=0.7, α_o=0.55). Restores orphan score variance from 0.0 (uniform) to 0.046.

**DNN Ensemble** (Filo-Priori-Ensemble): DeepOrder-inspired DNN + α-blending. α auto-optimized per project on validation set. DNN-only achieves APFD 0.686 on Industrial (-9.9% vs GNN), confirming GNN superiority when metadata is rich.

---

## Ablation Study (RQ2)

### Industrial Dataset (Filo-Priori-Full)

| Configuration | APFD | Delta | p-value |
|---|---|---|---|
| Full Model | 0.7611 | -- | -- |
| w/o Co-Failure Graph (GNN→MLP) | 0.6508 | -17.0% | <0.001*** |
| w/o Orphan KNN Scoring | 0.7145 | -5.9% | <0.001*** |
| w/o Dual-Balancing | 0.7291 | -4.0% | <0.001*** |
| w/o DeepOrder Features | 0.7443 | -2.0% | 0.003** |
| w/o Threshold Optimization | 0.7519 | -1.0% | 0.042* |

### RTPTorrent — 20 Projects (Filo-Priori-Ensemble)

| Configuration | APFD | Delta | Sig. |
|---|---|---|---|
| Full Ensemble (V14) | 0.8540 | -- | -- |
| w/o DNN Ensemble | 0.8322 | -2.6% | p<0.001*** |
| w/o GATv2 | 0.8451 | -1.0% | p<0.05* |
| w/o Semantic Stream | 0.8450 | -1.1% | ns |
| w/o Multi-Edge Graph | 0.8513 | -0.3% | ns |

**Cross-dataset insight:** On Industrial data, the co-failure graph and GATv2 attention dominate (+17.0%). On RTPTorrent, the DNN ensemble is the primary driver, though the V14 execution-level temporal GNN substantially reduced reliance on it (drop went from -13.1% to -2.6%). **Negative result:** Semantic stream provides no significant improvement on either dataset.

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

**Cross-dataset insight:** Both datasets show graceful degradation under temporal constraints with no concept drift. RTPTorrent's smaller drop (-2.7% vs -13-19%) reflects larger training sets in multi-project evaluation.

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
| α (GNN-DNN blend) | 0.0, 0.3, 0.5, 0.7, 1.0 | 0.0-0.7 (stable) | 13.1% (α=1.0 critical) |
| DNN Epochs | 5, 10, 15, 20 | 15-20 | 0.8% |
| Max pos_weight | 10, 25, 50, 100 | 50-100 | 1.6% |

**Cross-dataset insight:** Both datasets confirm robustness to continuous hyperparameters. Industrial max range: 3.6%. RTPTorrent max range: 1.6% (excl. degenerate α=1.0). The most impactful decision is architectural (including the DNN ensemble), not hyperparameter tuning.

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
| Access | Anonymized version in replication package |

### Dataset 2: RTPTorrent

| Statistic | Value |
|---|---|
| Projects | 20 Java projects |
| Total Builds | >100,000 |
| Builds with Failures | 2,937 |
| Source | Travis CI build logs (MSR 2020) |
| Semantic Info | Limited (test class/method names only) |
| License | CC BY 4.0 (publicly available) |

---

## Project Structure

```
filo-priori/
├── CLAUDE.md                        # AI agent instructions + frozen parameters
├── README.md                        # This file
├── main.py                          # Main entry point (Industrial dataset)
├── requirements.txt                 # Dependencies
├── configs/
│   └── experiment_industry_optimized_v3.yaml  # Industry config (APFD 0.7611) — FROZEN
├── experiments/
│   ├── run_filopriori_rtptorrent_v14.py       # Filo-Priori-Ensemble (APFD 0.8540) — FROZEN
│   ├── run_rtptorrent_ablation_sensitivity.py # Ablation + Sensitivity + Temporal CV
│   ├── run_dnn_ensemble_industry.py           # DNN ensemble verification on Industrial
│   ├── run_deeporder_*.py                     # DeepOrder baselines
│   ├── run_noderank_*.py                      # NodeRank baselines
│   ├── run_retecs_*.py                        # RETECS baselines
│   ├── run_tcpnet_*.py                        # TCP-Net baselines
│   ├── run_failrank_bb_*.py                   # FailRank-BB baselines
│   └── archived/                              # Old Filo-Priori versions (not in paper)
├── paper/
│   ├── main_ieee_tse.tex            # IEEE TSE manuscript (Regular Journal First)
│   ├── main_ieee_tse.pdf            # Compiled PDF
│   ├── references_ieee.bib          # Bibliography
│   ├── cover_letter.md              # Cover letter for ScholarOne
│   ├── novelty_statement.md         # 200-word Novelty Statement (Journal First)
│   ├── figures/                     # Paper figures (PDF)
│   ├── sections/                    # Paper sections (results, discussion, threats)
│   └── Computer_Society_LaTeX_template/  # Official IEEE CS template reference
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
│   ├── filopriori_rtptorrent_v14/          # RTPTorrent V14 results (APFD 0.8540)
│   ├── rtptorrent_ablation_sensitivity/    # RTPTorrent ablation + sensitivity + temporal CV
│   ├── dnn_ensemble_industry/              # DNN ensemble verification (APFD 0.686)
│   ├── deeporder_*/                        # DeepOrder baseline results
│   ├── noderank_*/                         # NodeRank baseline results
│   ├── retecs_*/                           # RETECS baseline results
│   ├── tcpnet_*/                           # TCP-Net baseline results
│   └── failrank_bb_*/                      # FailRank-BB baseline results
├── docs/                            # Technical documentation
└── cache/                           # Pre-computed embeddings and graphs
```

---

## Training Configuration

### Filo-Priori-Full (Industrial, APFD 0.7611)

| Parameter | Value |
|---|---|
| Framework | PyTorch 2.0, PyTorch Geometric 2.3 |
| Hardware | NVIDIA RTX 3090 (24 GB VRAM) |
| Optimizer | AdamW (weight_decay=1e-4) |
| Learning Rate | 3e-5 with cosine annealing |
| Batch Size | 16 |
| Epochs | 80 (early stop patience=15) |
| Loss | Focal Loss (α_f=0.75, γ=2.0) |
| Balanced Sampling | 29:1 (minority:majority) |
| GNN | GATv2, 1 layer, 2 heads, hidden_dim=128 |
| Monitoring | val_f1_macro |

### Filo-Priori-Ensemble (RTPTorrent, APFD 0.8540)

| Parameter | Value |
|---|---|
| Learning Rate | 1e-3 |
| Batch Size | 16 |
| Max Epochs | 30 (early stop patience=7) |
| GNN | GATv2, hidden_dim=128, 4 heads |
| DNN Architecture | [64, 32, 16] with sigmoid output |
| DNN Epochs | 15 (V15 sensitivity) |
| Max pos_weight (DNN) | 50.0 (clamp for rare-failure projects) |
| α-blending | Grid search over [0.0, 0.1, ..., 0.9] per project |
| Min val failure builds | 3 (guard against degenerate α) |

> **CRITICAL:** All parameters are FROZEN. See `CLAUDE.md` for the full configuration and the rationale behind each choice.

---

## Baselines

### Deep Learning Baselines

- **DeepOrder** (Chen et al., 2023): DNN with 8 historical features
- **NodeRank** (Li et al., 2024): Mutation-based analysis with ensemble learning
- **RETECS** (Spieker et al., 2017): Reinforcement learning for TCP in CI
- **TCP-Net** (Abdelkarim et al., 2022): End-to-end DNN with temporal execution features
- **FailRank-BB** (Hernandes et al., 2025): BERT embeddings + Logistic Regression

### Heuristic Baselines (Reference)

- **Random**: Random ordering (APFD ~ 0.5)
- **Recency**: Prioritizes recently failed tests
- **RecentFailureRate**: Failure rate in last 5 builds
- **FailureRate**: Overall historical failure rate
- **GreedyHistorical**: Multi-heuristic greedy selection

---

## Submission Package (IEEE TSE)

Materials prepared for IEEE TSE submission via ScholarOne (`mc.manuscriptcentral.com/tse-cs`):

| File | Purpose |
|---|---|
| [`paper/main_ieee_tse.tex`](paper/main_ieee_tse.tex) | Main manuscript (LaTeX source) |
| [`paper/main_ieee_tse.pdf`](paper/main_ieee_tse.pdf) | Compiled PDF |
| [`paper/references_ieee.bib`](paper/references_ieee.bib) | Bibliography |
| [`paper/cover_letter.md`](paper/cover_letter.md) | Cover letter for the Editor-in-Chief |
| [`paper/novelty_statement.md`](paper/novelty_statement.md) | 200-word Novelty Statement (Regular Journal First) |
| `paper/figures/` | 6 paper figures (PDF) |
| `paper/sections/` | Modular paper sections |

**Submission type:** Regular (Journal First) — eligible for presentation at the ICSE Journal-First track, at the journal's and ICSE's discretion.

---

## Documentation

| Document | Description |
|---|---|
| [CLAUDE.md](CLAUDE.md) | AI agent instructions, frozen parameters, valid experiments |
| [paper/main_ieee_tse.tex](paper/main_ieee_tse.tex) | Full IEEE TSE manuscript |
| [paper/cover_letter.md](paper/cover_letter.md) | Cover letter draft |
| [paper/novelty_statement.md](paper/novelty_statement.md) | Journal First Novelty Statement |

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| GPU VRAM | 8 GB | 12 GB+ |
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

See [`requirements.txt`](requirements.txt) for the complete pinned environment.

---

## Citation

```bibtex
@article{ribeiro2026filopriori,
  title   = {Filo-Priori: Co-Failure Graph Attention for Test Case
             Prioritization in Continuous Integration},
  author  = {Ribeiro, Acauan C. and Feitosa, Eduardo L. and
             Carvalho, Andre L. da Costa and Santos, Eulanda M. dos and
             Gadelha, Bruno F. and Soares, Yan R. and Nascimento, Jose},
  journal = {IEEE Transactions on Software Engineering},
  year    = {2026},
  note    = {Submitted (Regular Journal First)}
}
```

---

## References

- **GAT**: Veličković, P., et al. (2018). Graph Attention Networks. ICLR.
- **GATv2**: Brody, S., et al. (2022). How Attentive are Graph Attention Networks? ICLR.
- **Focal Loss**: Lin, T., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
- **SBERT**: Reimers, N., and Gurevych, I. (2019). Sentence-BERT. EMNLP.
- **DeepOrder**: Chen, J., et al. (2023). Deep Learning for TCP in CI Testing. TSE.
- **NodeRank**: Li, Z., et al. (2024). NodeRank: Test Input Prioritization for GNNs.
- **RTPTorrent**: Mattis, T., et al. (2020). RTPTorrent: An Open-Source Dataset. MSR.
- **FailRank-BB**: Hernandes, V., et al. (2025). A Method for Regression Testing Plan Ordering.

---

## Acknowledgments

This work was partially supported by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) — Finance Code 001, and by Motorola Mobility Comércio de Produtos Eletrônicos Ltda., under the auspices of the Brazilian Federal Law No. 8.387/1991, which also provided access to the industrial dataset used in this study.

**Conflicts of Interest:** Author Jose Nascimento is employed by Motorola Mobility Comércio de Produtos Eletrônicos Ltda. The remaining authors declare no competing interests. Motorola Mobility had no role in study design, data analysis, interpretation of results, or the decision to submit this manuscript for publication.

---

## License

- **Code:** Released under permissive academic-use terms for the purposes of replication and validation of the IEEE TSE paper.
- **RTPTorrent dataset:** CC BY 4.0 (original authors).
- **Industrial QTA dataset:** Anonymized version in the replication package; raw industrial data is proprietary to Motorola Mobility under the Brazilian Federal Law No. 8.387/1991.

For licensing inquiries beyond academic replication, contact the corresponding author: acauan.ribeiro@icomp.ufam.edu.br
