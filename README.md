# Filo-Priori: Co-Failure Graph Attention for Test Case Prioritization in Continuous Integration

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyG](https://img.shields.io/badge/PyG-2.3+-orange.svg)
![APFD-Industrial](https://img.shields.io/badge/APFD%20Industrial-0.761-brightgreen.svg)
![APFD-RTPTorrent](https://img.shields.io/badge/APFD%20RTPTorrent-0.854-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Submitted%20to%20IEEE%20TSE-blue.svg)

A **graph-based deep learning framework** for Test Case Prioritization (TCP) in CI/CD pipelines. Centers on **GATv2 graph attention over a co-failure test relationship graph**, complemented by a DNN ensemble with data-driven α-blending for semantically-sparse settings.

**Target Journal:** IEEE Transactions on Software Engineering (IEEE TSE)
**Submission Type:** Regular (Journal First)
**Replication Package:** https://github.com/filotransformer/filopriori
**Corresponding Author:** Acauan C. Ribeiro (acauan.ribeiro@icomp.ufam.edu.br)

**Authors:** Acauan C. Ribeiro, Eduardo L. Feitosa, Andre L. da Costa Carvalho, Eulanda M. dos Santos, Bruno F. Gadelha, Yan R. Soares, Jose Carlos Rangel do Nascimento, Nícolas Riccieri Gardin Assumpção

**Affiliations:** Instituto de Computação (IComp), Universidade Federal do Amazonas (UFAM) · Motorola Mobility Comércio de Produtos Eletrônicos Ltda.

---

## Abstract

In Continuous Integration (CI), growing test suites make exhaustive testing impractical, motivating TCP to maximize early fault detection. Existing approaches treat tests as independent entities, ignoring structural relationships. **Filo-Priori** models inter-test relationships through GATv2 over a co-failure graph, with a DNN ensemble for sparse-metadata settings. The framework provides two configurations:

1. **Filo-Priori-Full** (metadata-rich): GATv2 over a multi-edge graph + KNN orphan imputation.
2. **Filo-Priori-Ensemble** (metadata-sparse): GATv2 over co-failure graph + DeepOrder-inspired DNN with validation-optimized α-blending.

**Key findings:** (i) co-failure edges are the decisive graph component (+17.0%, p<0.001); (ii) generic Sentence-BERT embeddings provide no significant improvement (p=0.309); (iii) dual-balancing (sampling + focal loss) prevents mode collapse.

---

## Results

| Dataset | Variant | APFD | vs. Best Baseline |
|---|---|---|---|
| Industrial QTA (277 builds) | Filo-Priori-Full | **0.761** | +10.2% (DeepOrder, p<0.001) |
| RTPTorrent (20 Java projects, 2,937 builds) | Filo-Priori-Ensemble | **0.854** | +1.6% (TCP-Net, ns after Holm-Bonferroni) |

### Industrial Dataset — Significant gains over all 5 baselines (Bonferroni-corrected)

| Method | APFD | Δ |
|---|---|---|
| **Filo-Priori-Full** | **0.761** | -- |
| DeepOrder | 0.689 | +10.2% (p<0.001) |
| TCP-Net | 0.670 | +13.3% (p<0.001) |
| NodeRank | 0.661 | +14.9% (p<0.001) |
| RETECS | 0.641 | +18.6% (p<0.001) |
| FailRank-BB | 0.595 | +27.6% (p<0.001) |

### RTPTorrent — Highest numerical APFD; only 2 baselines significantly worse after correction

| Method | APFD | Δ | Holm-Bonferroni |
|---|---|---|---|
| **Filo-Priori-Ensemble** | **0.854** | -- | -- |
| TCP-Net | 0.825 | +1.6% | ns |
| FailRank-BB | 0.822 | +2.0% | ns |
| DeepOrder | 0.814 | +3.0% | marginal |
| NodeRank | 0.809 | +5.6% | **sig** |
| RETECS | 0.679 | +23.5% | **sig** |

---

## Research Questions (Summary)

| RQ | Finding |
|---|---|
| **RQ1** — Effectiveness | Industrial: +10.2% to +27.6% over all baselines (p<0.001). RTPTorrent: highest aggregate APFD; robust over NodeRank/RETECS after correction |
| **RQ2** — Components | Industrial: GATv2 co-failure graph dominates (+17.0%). RTPTorrent: DNN ensemble is primary driver (-2.6% without it). Semantic stream non-significant on both |
| **RQ3** — Temporal Robustness | Industrial: 13–19% degradation under temporal CV. RTPTorrent: -2.7% only. No concept drift |
| **RQ4** — Hyperparameter Sensitivity | Industrial: max 3.6%. RTPTorrent: max 1.6% (excl. degenerate α=1.0). Robust |

Full details, ablations, and sensitivity tables are in the paper (`paper/main_ieee_tse.tex`).

---

## Quick Start

The **RTPTorrent** dataset is public (CC BY 4.0) and is the recommended path to reproduce the paper's open-source results. The **Industrial QTA** dataset is proprietary and **cannot be redistributed** (see [Datasets](#datasets) below).

### 1. Environment setup

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download and preprocess RTPTorrent (public dataset)

Full download and preprocessing instructions are in [`datasets/02_rtptorrent/README.md`](datasets/02_rtptorrent/README.md).

```bash
# Download (~4.1 GB from Zenodo) + preprocess
python scripts/preprocessing/download_rtptorrent.py
python scripts/preprocessing/preprocess_rtptorrent.py
```

### 3. Reproduce RTPTorrent results

```bash
# Main result (Filo-Priori-Ensemble, V14) — APFD 0.8540
python experiments/run_filopriori_rtptorrent_v14.py

# Ablation + sensitivity + temporal CV (~14h)
python experiments/run_rtptorrent_ablation_sensitivity.py --all

# Baselines on RTPTorrent
python experiments/run_deeporder_rtptorrent.py
python experiments/run_tcpnet_rtptorrent.py
python experiments/run_noderank_rtptorrent.py
python experiments/run_retecs_rtptorrent.py
python experiments/run_failrank_bb_rtptorrent.py
```

### 4. Industrial dataset (proprietary, internal use only)

```bash
# Requires the proprietary Motorola/QTA CSVs in datasets/01_industry/
python main.py --config configs/experiment_industry_optimized_v3.yaml
```

---

## Architecture (Summary)

```
Co-Failure Graph (CORE)  ──►  GATv2 Structural Stream  ┐
                                                       ├──►  α-blending  ──►  Ranked Tests
DNN Ensemble (Filo-Priori-Ensemble)                    ┘
   (DeepOrder-inspired, validation-optimized α)

Optional: Semantic Stream (SBERT) + Cross-Attention Fusion
          → no significant benefit (p=0.309); kept exploratory
Optional: 4 supplementary edge types (Co-Success, Component, Semantic, Temporal)
          → no aggregate improvement over co-failure alone
Filo-Priori-Full: + KNN orphan imputation (k=5, T=0.7, α_o=0.55)
```

**Core configuration:** GATv2 with 1 layer, 2 heads, hidden_dim=128. 19 structural features (10 base + 9 DeepOrder-inspired). Focal Loss (α_f=0.75, γ=2.0) + balanced sampling (29:1). See the paper for the complete architecture.

---

## Datasets

| Dataset | Builds | With Failures | Tests | Pass:Fail | Access |
|---|---|---|---|---|---|
| Industrial QTA | 1,339 | 277 (20.7%) | 2,347 | 37:1 | **Proprietary — not downloadable** |
| RTPTorrent (20 Java projects) | >100,000 | 2,937 | -- | varies | **CC BY 4.0 — public download** |

### Industrial QTA (proprietary)

Real test execution data from a Motorola mobile-device CI/CD pipeline, collected through the Qodo Test Automation (QTA) system. This dataset is **commercial and confidential**, governed by Brazilian Federal Law No. 8.387/1991 (SUFRAMA), and **cannot be redistributed**. It is used in the paper only for internal validation. Field schema and statistics are documented in [`datasets/01_industry/README.md`](datasets/01_industry/README.md).

External readers cannot reproduce the Industrial experiments without access to the proprietary CSV files. All other paper artifacts (RTPTorrent results, ablation, sensitivity, temporal CV) are fully reproducible from the public dataset.

### RTPTorrent (public, CC BY 4.0)

Open-source dataset from MSR 2020 (Mattis et al.) with >100,000 Travis CI build logs across 20 Java projects on GitHub. The dataset is hosted on Zenodo and can be downloaded directly:

- **Zenodo:** https://zenodo.org/records/3712290
- **Paper:** https://doi.org/10.1145/3379597.3387458
- **Local instructions:** [`datasets/02_rtptorrent/README.md`](datasets/02_rtptorrent/README.md) — covers download, preprocessing, expected directory layout, and how to run Filo-Priori-Ensemble on it.

---

## Project Structure

```
filo-priori/
├── README.md
├── main.py                                          # Industrial entry point
├── requirements.txt
├── configs/
│   └── experiment_industry_optimized_v3.yaml        # Industrial config — FROZEN
├── experiments/
│   ├── run_filopriori_rtptorrent_v14.py             # Filo-Priori-Ensemble — FROZEN
│   ├── run_rtptorrent_ablation_sensitivity.py       # Ablation/sensitivity/temporal CV
│   ├── run_dnn_ensemble_industry.py                 # DNN-only verification (Industrial)
│   ├── run_{deeporder,noderank,retecs,tcpnet,failrank_bb}_*.py  # Baselines
│   └── archived/                                    # Old versions (not in paper)
├── paper/
│   ├── main_ieee_tse.tex                            # IEEE TSE manuscript
│   ├── main_ieee_tse.pdf
│   ├── references_ieee.bib
│   ├── cover_letter.md                              # For ScholarOne
│   ├── novelty_statement.md                         # 200-word Journal First statement
│   ├── figures/                                     # 6 paper figures (PDF)
│   └── sections/                                    # Modular paper sections
├── src/                                             # Models, layers, baselines, evaluation
├── datasets/                                        # 01_industry/, 02_rtptorrent/
├── results/                                         # Per-experiment outputs
└── docs/                                            # Technical documentation
```

---

## Submission Package (IEEE TSE)

| File | Purpose |
|---|---|
| `paper/main_ieee_tse.tex` / `.pdf` | Main manuscript |
| `paper/references_ieee.bib` | Bibliography |
| `paper/cover_letter.md` | Cover letter for the Editor-in-Chief |
| `paper/novelty_statement.md` | 200-word Novelty Statement (Regular Journal First) |
| `paper/figures/` | 6 paper figures (PDF) |

Submitted via **ScholarOne** at https://mc.manuscriptcentral.com/tse-cs as **Regular (Journal First)**. TSE uses **single-anonymous** review (no anonymization required).

---

## Requirements

- Hardware: 16 GB RAM (min) / 32 GB (recommended), GPU with 8 GB+ VRAM, CUDA 11.8+
- Software: PyTorch ≥2.0, PyTorch Geometric ≥2.3, sentence-transformers, pandas, scikit-learn. See `requirements.txt`.

---

## Citation

```bibtex
@article{ribeiro2026filopriori,
  title   = {Filo-Priori: Co-Failure Graph Attention for Test Case
             Prioritization in Continuous Integration},
  author  = {Ribeiro, Acauan C. and Feitosa, Eduardo L. and
             Carvalho, Andre L. da Costa and Santos, Eulanda M. dos and
             Gadelha, Bruno F. and Soares, Yan R. and
             Nascimento, Jose Carlos Rangel do and
             Assump{\c{c}}{\~a}o, N{\'i}colas Riccieri Gardin},
  journal = {IEEE Transactions on Software Engineering},
  year    = {2026},
  note    = {Submitted (Regular Journal First)}
}
```

---

## Acknowledgments

This work was partially supported by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) — Finance Code 001, and by Motorola Mobility Comércio de Produtos Eletrônicos Ltda., under the auspices of the Brazilian Federal Law No. 8.387/1991, which also provided access to the industrial dataset used in this study.

**Conflicts of Interest:** Authors Jose Carlos Rangel do Nascimento and Nícolas Riccieri Gardin Assumpção are employed by Motorola Mobility. The remaining authors declare no competing interests. Motorola Mobility had no role in study design, data analysis, interpretation of results, or the decision to submit this manuscript.

---

## License

- **Code:** Academic-use terms for replication and validation of the IEEE TSE paper.
- **RTPTorrent dataset:** CC BY 4.0 (original authors).
- **Industrial QTA dataset:** Proprietary and not redistributable (NDA, Brazilian Federal Law No. 8.387/1991); not included in this replication package. The industrial experiments are not externally reproducible.

Licensing inquiries beyond academic replication: acauan.ribeiro@icomp.ufam.edu.br
