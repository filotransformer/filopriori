# Filo-Priori - Paper Submission Materials

## Paper Title

**Filo-Priori: A Multi-Edge Graph Attention Approach to Test Case Prioritization**

Target Journal: IEEE Transactions on Software Engineering (IEEE TSE)

## Authors

Acauan C. Ribeiro, Eduardo L. Feitosa, Andre L. da Costa Carvalho, Eulanda M. dos Santos, Bruno F. Gadelha, Yan R. Soares, Jose Carlos Rangel do Nascimento, and Nícolas Riccieri Gardin Assumpção

## Directory Structure

```
paper/
├── main_ieee_tse.tex          # Main paper (IEEE TSE format)
├── main_ieee_tse.pdf          # Compiled PDF
├── references_ieee.bib        # Bibliography
├── figures/                   # All figures (PDF)
│   ├── fig_framework_overview_new.pdf
│   ├── fig_dual_stream_architecture_new.pdf
│   ├── fig_multi_edge_graph.pdf
│   ├── fig_orphan_pipeline_new.pdf
│   ├── fig_apfd_comparison.pdf
│   └── fig_ablation_crossdataset.pdf
└── sections/                  # Paper sections
    ├── results_ieee.tex       # RQ1-RQ4 results
    ├── discussion_ieee.tex    # Discussion
    └── threats_ieee.tex       # Threats to validity
```

## Compilation

```bash
cd paper/
pdflatex main_ieee_tse.tex
bibtex main_ieee_tse
pdflatex main_ieee_tse.tex
pdflatex main_ieee_tse.tex
```

## Key Results

### Summary Across Two Datasets

| Experiment | Variant | APFD | vs. Best Baseline |
|---|---|---|---|
| Industrial QTA | Filo-Priori-Full | **0.761** | +10.2% (DeepOrder, p<0.001) |
| RTPTorrent (20 projects) | Filo-Priori-Ensemble | **0.854** | +1.6% (TCP-Net, ns after correction) |

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

After Holm-Bonferroni correction, only NodeRank and RETECS remain significant. Narrow margins over TCP-Net, FailRank-BB, and DeepOrder should be interpreted with caution.

## Key Findings

1. **GATv2 graph attention is the dominant contributor** (+17.0% in component isolation, p<0.001)
2. **Negative result on semantic features**: Generic Sentence-BERT embeddings provide no statistically significant improvement (p=0.309 industrial, Δ=0.0% on 19/20 RTPTorrent projects)
3. **DNN ensemble is critical for sparse-metadata settings**: Primary driver on RTPTorrent
4. **Shallow GNN architectures suffice**: 1 layer, 2 heads outperforms deeper configurations
5. **Simplified dual-balancing prevents mode collapse**: Balanced sampling (29:1) + focal loss (α_f=0.75, γ=2.0)

## Research Questions

| RQ | Question | Key Finding |
|---|---|---|
| RQ1 | Effectiveness? | Industrial: APFD 0.761, all baselines beaten (p<0.001). RTPTorrent: APFD 0.854, highest aggregate; NodeRank/RETECS robust after correction |
| RQ2 | Component contributions? | Industrial: Graph (+10.0%), Orphan KNN (+5.9%), Balancing (+4.0%). RTPTorrent: DNN ensemble (-2.6%, p<0.001). Semantic stream: not significant on either dataset |
| RQ3 | Temporal robustness? | Industrial: 0.619-0.663 (13-19% degradation, expected). RTPTorrent: 0.816 (-2.7%), no concept drift |
| RQ4 | Hyperparameter sensitivity? | Industrial: max 3.6%. RTPTorrent: max 1.6% (excl. α=1.0). Robust to continuous hyperparameters |

## Paper Sections

| Section | Content |
|---|---|
| 1. Introduction | Problem, contributions (incl. negative result on semantics), RQs |
| 2. Background | TCP, APFD, GAT/GATv2, Focal Loss, Cross-Attention |
| 3. Related Work | Traditional, ML/DL, and Graph-based TCP; positioning table |
| 4. Approach | Two variants: Filo-Priori-Full and Filo-Priori-Ensemble |
| 5. Experimental Design | Datasets, baselines, metrics, implementation details |
| 6. Results | RQ1-RQ4 with cross-dataset synthesis |
| 7. Discussion | Structural dominance, negative result, failure analysis, deployment |
| 8. Threats to Validity | Internal, external, construct, conclusion, reproducibility |
| 9. Conclusion | Summary, four actionable insights, future work |

---

Updated: 2026-03
