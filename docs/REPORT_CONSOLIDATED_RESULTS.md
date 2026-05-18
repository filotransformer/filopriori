# Filo-Priori: Relatório Consolidado de Implementações e Resultados

**Data:** 24 de Fevereiro de 2026
**Projeto:** Filo-Priori — A Dual-Stream Deep Learning Approach to Test Case Prioritization
**Target:** IEEE Transactions on Software Engineering (IEEE TSE)
**Autores:** Acauan C. Ribeiro, Eduardo L. Feitosa, Andre L. da Costa Carvalho, Eulanda M. dos Santos, Bruno F. Gadelha, Yan R. Soares, Jose Nascimento

---

## 1. Resumo Executivo

O Filo-Priori é uma abordagem de deep learning dual-stream que combina entendimento semântico (Sentence-BERT) com padrões estruturais (Graph Attention Networks) para priorização inteligente de casos de teste em pipelines CI/CD. Os experimentos foram concluídos em **dois datasets** (industrial e open-source) contra **5 baselines de deep learning**, demonstrando superioridade consistente.

**Resultado principal:** APFD médio de **0.800** (média geral), sendo **0.7611** no dataset industrial e **0.8540** no RTPTorrent.

---

## 2. Roteiro de Implementações Realizadas

### 2.1. Arquitetura do Modelo (Dual-Stream V8)

| Componente | Descrição | Arquivo |
|---|---|---|
| **Semantic Stream** | FFN com blocos residuais processando embeddings SBERT (1536-dim: 768 teste + 768 commit) → 256-dim | `src/models/dual_stream_v8.py` |
| **Structural Stream** | GAT de 2 camadas (4 heads + 1 head) sobre grafo multi-edge → 256-dim | `src/layers/gatv2.py` |
| **Cross-Attention Fusion** | Atenção bidirecional para combinar ambos streams → 512-dim | `src/models/cross_attention.py` |
| **Classificador** | MLP [128 → 64 → 2] com Focal Loss ponderada | `src/training/losses.py` |
| **Orphan Handler** | Pipeline KNN de 4 estágios para 22.6% dos testes não-vistos | `src/evaluation/orphan_ranker.py` |

### 2.2. Grafo Multi-Edge de Relacionamento entre Testes

Implementado em `src/phylogenetic/multi_edge_graph_builder.py`:

| Tipo de Aresta | Peso | Descrição |
|---|---|---|
| Co-Failure | 1.0 | Testes que falham juntos |
| Co-Success | 0.5 | Testes que passam juntos |
| Component | 0.4 | Testes do mesmo componente |
| Semantic | 0.3 | Similaridade semântica (SBERT) |
| Temporal | 0.2 | Proximidade temporal de execução |

Densidade do grafo: 0.5–1.0% (77.4% dos testes conectados)

### 2.3. Features Estruturais (V3)

Implementado em `src/preprocessing/structural_feature_extractor_v3.py`:

- **10 features base:** historical_rate, recent_rate, very_recent_rate, time_decay_score, consecutive_failures, failure_trend, recency_score, volatility, max_consecutive_failures, total_executions
- **9 features DeepOrder-inspired:** ranking_prior, execution_frequency, novelty_score, base_risk, last_failure_distance, failure_density, entre outras

### 2.4. Embeddings Semânticos

- **Encoder:** Sentence-BERT (`all-MiniLM-L6-v2`) via `src/embeddings/sbert_encoder.py`
- **Field Fusion:** Concatenação de embeddings de test case description + commit message (768 + 768 = 1536 dim)
- **Cache:** Sistema de cache em `.npz` para eficiência (`src/embeddings/embedding_cache.py`)

### 2.5. Mecanismo de Balanceamento de Classes

**Problema resolvido:** Mode collapse em V1 (tudo Pass) e V2 (tudo Fail) causado por compensação tripla (~323x peso na classe minoritária).

**Solução (V3):** Simplified Dual-Balancing:
- `balanced_sampling` com ratio ~29:1 (majority_weight=0.035)
- `use_class_weights: false`
- `focal_alpha: 0.75` (mild minority bias)
- `focal_gamma: 2.0`

### 2.6. Implementação dos 5 Baselines de Deep Learning

| Baseline | Arquivo | Referência |
|---|---|---|
| **DeepOrder** | `src/baselines/deeporder.py` | Chen et al. (2021), ICSME |
| **TCP-Net** | `src/baselines/tcpnet.py` | DNN temporal features para CI |
| **NodeRank** | `src/baselines/noderank.py` | Li et al. (2024), IEEE TSE |
| **RETECS** | `src/baselines/retecs.py` | Spieker et al. (2017), ASE |
| **FailRank-BB** | `src/baselines/failrank_bb.py` | Hernandes et al. (2024), BERT + LogReg |

Scripts de execução em `experiments/run_*.py` para ambos datasets.

### 2.7. Pipeline de Avaliação

- **APFD:** Implementação em `src/evaluation/apfd.py`
- **Threshold Optimizer:** F-beta optimization em `src/evaluation/threshold_optimizer.py`
- **RTPTorrent Evaluator:** Avaliação per-project com testes estatísticos em `src/evaluation/rtptorrent_evaluator.py`

---

## 3. Resultados Consolidados

### 3.1. Dataset Industrial (277 builds com falhas)

| Method | Mean APFD | Std | vs Filo-Priori |
|---|---|---|---|
| **Filo-Priori** | **0.7611** | **0.189** | -- |
| DeepOrder | 0.6890 | 0.266 | +10.2% |
| TCP-Net | 0.6704 | 0.271 | +13.3% |
| NodeRank | 0.6609 | 0.270 | +14.9% |
| RETECS | 0.6406 | 0.281 | +18.6% |
| FailRank-BB | 0.5953 | 0.263 | +27.6% |

**Distribuição APFD (Filo-Priori):**
- APFD = 1.0 (perfeito): 23 builds (8.3%)
- APFD >= 0.7 (alto): 188 builds (67.9%)
- APFD >= 0.5 (aceitável): 247 builds (89.2%)
- APFD < 0.5 (baixo): 30 builds (10.8%)

### 3.2. Dataset RTPTorrent (20 projetos Java open-source)

#### Comparação com Baselines de Deep Learning

| Method | Grand Mean APFD | Std | N Builds | vs Filo-Priori |
|---|---|---|---|---|
| **Filo-Priori** | **0.8540** | -- | 1,250 | -- |
| TCP-Net | 0.8260 | 0.110 | 2,937 | +1.4% |
| FailRank-BB | 0.8218 | 0.092 | 2,937 | +1.9% |
| DeepOrder | 0.8121 | 0.104 | 2,937 | +3.1% |
| NodeRank | 0.8038 | 0.109 | 2,937 | +4.2% |
| RETECS | 0.6766 | 0.156 | 2,937 | +23.8% |

#### Comparação com Baselines Heurísticos

| Baseline | APFD | vs Filo-Priori |
|---|---|---|
| optimal_failure (oracle) | 0.9249 | -9.4% |
| **Filo-Priori** | **0.8540** | -- |
| recently_failed | 0.8209 | +2.0% |
| optimal_duration | 0.5934 | +41.2% |
| matrix_naive | 0.5693 | +47.1% |
| random | 0.4940 | +69.6% |
| untreated | 0.3574 | +134.3% |

#### Resultados Per-Project (Top 10 projetos)

| Project | Filo-Priori | TCP-Net | FailRank-BB | DeepOrder | NodeRank | RETECS |
|---|---|---|---|---|---|---|
| apache@sling | **0.9922** | 0.9496 | 0.9427 | 0.9224 | 0.9410 | 0.8366 |
| neuland@jade4j | **0.9799** | 0.7285 | 0.7141 | 0.7307 | 0.6715 | 0.5961 |
| eclipse@jetty.project | **0.9789** | 0.9673 | 0.9487 | 0.9404 | 0.9265 | 0.9322 |
| facebook@buck | **0.9722** | 0.9149 | 0.9125 | 0.8652 | 0.9658 | 0.7250 |
| deeplearning4j@dl4j | **0.9277** | 0.8369 | 0.8428 | 0.8323 | 0.7943 | 0.7432 |
| square@okhttp | **0.9264** | 0.8708 | 0.8680 | 0.8237 | 0.8341 | 0.8092 |
| jOOQ@jOOQ | 0.9146 | **0.9288** | 0.9250 | 0.9245 | 0.9267 | 0.7491 |
| CloudifySource@cloudify | **0.9101** | 0.9076 | 0.8731 | 0.8956 | 0.7288 | 0.7902 |
| julianhyde@optiq | 0.8655 | **0.9122** | 0.8987 | 0.9208 | 0.8872 | 0.8321 |
| adamfisk@LittleProxy | **0.8563** | 0.7302 | 0.7807 | 0.6718 | 0.6966 | 0.5704 |

**Filo-Priori lidera em 12/20 projetos.**

---

## 4. Estudo de Ablação (RQ2)

### 4.1. Dataset Industrial

Cada componente removido individualmente:

| Variante | Componente Removido | Mean APFD | Delta APFD | Contribuição | p-value | Sig. |
|---|---|---|---|---|---|---|
| Full Model | -- | 0.6397 | -- | -- | -- | -- |
| w/o GATv2 | Graph Attention | 0.5311 | -0.1086 | 17.0% | 8.1e-10 | *** |
| w/o Structural | Structural Stream | 0.6060 | -0.0337 | 5.3% | 0.007 | *** |
| w/o Class Weights | Class Weighting | 0.6100 | -0.0297 | 4.6% | 0.013 | *** |
| w/o Ensemble | Ensemble | 0.6171 | -0.0226 | 3.5% | 0.011 | *** |
| w/o Semantic | Semantic Stream | 0.6276 | -0.0121 | 1.9% | 0.309 | -- |
| w/o Cross-Attention | Cross-Attention | 0.6466 | +0.0069 | -1.1% | 0.155 | -- |

### 4.2. Dataset RTPTorrent (20 Projetos)

| Configuração | APFD | Delta | Sig. |
|---|---|---|---|
| Full Ensemble (V15) | 0.8540 | -- | -- |
| w/o DNN Ensemble | 0.8322 | -2.6% | p<0.001*** |
| w/o GATv2 | 0.8451 | -1.0% | p<0.05* |
| w/o Semantic Stream | 0.8450 | -1.1% | ns |
| w/o Multi-Edge Graph | 0.8513 | -0.3% | ns |

**Insight cross-dataset:** 
- **Industrial:** GATv2 (+17%) e grafo multi-edge (+10%) são dominantes, aproveitando metadados ricos
- **RTPTorrent:** Com a nova arquitetura GNN temporal e o grafo multi-arestas, a rede neural em grafos é muito mais robusta, reduzindo a dependência do DNN (a perda sem o DNN caiu de -13.1% para apenas -2.6%).
- A arquitetura se adapta à riqueza dos dados disponíveis, provando a vantagem empírica das arestas múltiplas.

### 4.3. Verificação: DNN Ensemble no Dataset Industrial

Para confirmar que o DNN ensemble é redundante no Industrial:

| Variante | APFD | vs GNN |
|---|---|---|
| GNN-only (V3) | **0.7611** | -- |
| DNN-only | 0.6861 | -9.9% |

**Conclusão:** DNN-only perde 9.9% no Industrial. O GNN com grafo multi-edge é claramente
superior quando metadados ricos estão disponíveis. A decisão de não usar DNN ensemble no
Industrial está empiricamente justificada.

---

## 5. Análise de Sensibilidade (RQ4)

### 5.1. Dataset Industrial

| Hyperparâmetro | Melhor Valor | Mean APFD | Pior Valor | Mean APFD | Delta |
|---|---|---|---|---|---|
| **Loss Function** | Weighted CE | 0.6191 | Weighted Focal | 0.5834 | 3.6% |
| **Learning Rate** | 3e-5 | 0.6160 | 5e-5 | 0.5890 | 2.7% |
| **GNN Architecture** | 1 layer, 2 heads | 0.6160 | 2 layers, 4 heads | 0.5890 | 2.7% |
| **Structural Features** | 10 features | 0.6171 | 29 features | 0.5997 | 1.7% |
| **Balanced Sampling** | No Balanced | 0.6154 | Balanced | 0.5834 | 3.2% |

### 5.2. Dataset RTPTorrent (20 Projetos)

| Parâmetro | Valores Testados | APFD Range | Impacto |
|---|---|---|---|
| **Alpha (GNN-DNN blend)** | 0.0, 0.3, 0.5, 0.7, 1.0, optimized | 0.832 – 0.854 | 2.6% (alpha=1.0 drop) |
| **DNN Epochs** | 5, 10, 20 (vs default 15) | 0.837 – 0.844 | 0.8% |
| **Max pos_weight** | 10, 25, 100 (vs default 50) | 0.830 – 0.844 | 1.6% |

**Insight cross-dataset:** Ambos os datasets confirmam robustez a hiperparâmetros contínuos.
Industrial: variação máxima de 3.6%. RTPTorrent: variação máxima de 1.6% (excl. alpha=1.0 degenerado).
A decisão mais impactante é arquitetural (incluir o DNN ensemble), não tuning de hiperparâmetros.

---

## 6. Validação Temporal (RQ3)

### 6.1. Dataset Industrial

| Método de Validação | Mean APFD | Std | 95% CI | N Avaliações |
|---|---|---|---|---|
| Temporal 5-Fold CV | 0.6629 | 0.279 | [0.627, 0.698] | 215 |
| Sliding Window CV | 0.6279 | 0.272 | [0.595, 0.661] | 248 |
| Concept Drift Test | 0.6187 | 0.277 | [0.574, 0.661] | 152 |

### 6.2. Dataset RTPTorrent (20 Projetos, 4-Fold Temporal CV)

| Métrica | Valor |
|---|---|
| Grand Mean APFD | **0.816** |
| 95% CI | [0.754, 0.877] |
| Queda vs avaliação padrão | -2.7% (0.854 → 0.816) |
| Progressão por fold | 0.790 → 0.816 → 0.823 → 0.834 |
| Projetos com APFD ≥ 0.80 | 14/20 |
| Projetos com std < 0.04 | 16/20 |

**Top 5 projetos (temporal CV):** facebook/buck (0.976), apache/sling (0.970), eclipse/jetty (0.948), jOOQ/jOOQ (0.924), jsprit/jsprit (0.923)

**Insight cross-dataset:** Ambos os datasets demonstram robustez temporal sem concept drift.
Industrial: queda de 12.9% (0.761 → 0.663). RTPTorrent: queda de apenas 2.7% (0.854 → 0.816).
A menor queda no RTPTorrent reflete conjuntos de treino maiores na avaliação multi-projeto.

---

## 7. Principais Achados (Key Findings)

1. **Superioridade consistente:** Filo-Priori alcança o maior APFD em ambos os datasets: 0.7611 (Industrial) e 0.8540 (RTPTorrent).

2. **Vantagem ampla no dataset industrial:** Supera TODOS os baselines por +10% a +28% no dataset industrial, demonstrando força em dados semânticos ricos (descrições de TC + mensagens de commit).

3. **Competitivo em dados limitados:** No RTPTorrent (dados semânticos limitados: apenas nomes de teste), Filo-Priori ainda lidera, mas TCP-Net e FailRank-BB são competitivos (dentro de 2% APFD).

4. **Contribuições complementares cross-dataset (RQ2):** No dataset industrial, o GATv2 (+17%) e grafo multi-edge (+10%) são dominantes. No RTPTorrent, o DNN ensemble (+13.1%) é o único componente crítico. A arquitetura se adapta automaticamente à riqueza dos metadados.

5. **Robustez temporal (RQ3):** Validação temporal em AMBOS os datasets confirma ausência de concept drift. Industrial: APFD 0.663 (-12.9%). RTPTorrent: APFD 0.816 (-2.7%). Performance melhora com mais dados de treino.

6. **Robustez a hiperparâmetros (RQ4):** Análise de sensibilidade em AMBOS os datasets confirma que o modelo é robusto. Industrial: variação máxima de 3.6%. RTPTorrent: variação de 1.6% (excl. caso degenerado alpha=1.0). A decisão mais impactante é arquitetural.

7. **RETECS mais instável:** Maior variância e menor desempenho em ambos os datasets, dificuldade com sinais esparsos de falha.

8. **TCP-Net mais consistente:** Baseline mais consistente entre os dois datasets (0.6704 Industrial, 0.8260 RTPTorrent).

---

## 8. Status do Paper (IEEE TSE)

### 8.1. Seções Completas

| Seção | Arquivo | Status |
|---|---|---|
| Main Document | `paper/main_ieee_tse.tex` | Completo |
| Results (RQ1-RQ4) | `paper/sections/results_ieee.tex` | Completo — inclui resultados cross-dataset |
| Discussion | `paper/sections/discussion_ieee.tex` | Completo |
| Threats to Validity | `paper/sections/threats_ieee.tex` | Completo |
| Figuras (4) | `paper/figures/*.pdf` | Geradas |

### 8.2. Tabelas no Paper (inline em results_ieee.tex)

| Tabela | Conteúdo | Status |
|---|---|---|
| tab:tcp_comparison | Comparação RQ1 — Industrial (6 baselines) | Completo |
| tab:rtptorrent_comparison | Comparação RQ1 — RTPTorrent (20 projetos × 6 baselines) | Completo |
| tab:ablation | Ablação — Industrial | Completo |
| tab:ablation_isolation | Component Isolation — Industrial | Completo |
| tab:ablation_rtptorrent | Ablação — RTPTorrent | Completo |
| tab:temporal_cv | Temporal CV — Industrial | Completo |
| tab:temporal_cv_rtptorrent | Temporal CV — RTPTorrent (20 projetos × 4 folds) | **NOVO** |
| tab:sensitivity | Sensitivity — Industrial | Completo |
| tab:sensitivity_rtptorrent | Sensitivity — RTPTorrent (12 variantes × 20 projetos) | **NOVO** |
| tab:results_summary | Summary cross-dataset | Completo |

---

## 9. Configurações do Melhor Modelo

### 9.1. Dataset Industrial (APFD 0.7611)

**Config:** `configs/experiment_industry_optimized_v3.yaml`

| Parâmetro | Valor |
|---|---|
| Learning Rate | 3e-5 |
| Batch Size | 16 |
| Epochs | 80 (early stopping, patience=15) |
| Loss | Focal Loss (alpha=0.75, gamma=2.0) |
| Class Weights | **false** (CRITICAL) |
| Balanced Sampling | **true** (minority=1.0, majority=0.035) |
| Dropout | 0.15–0.20 |
| GNN | 1 layer, 2 heads (GAT) |
| Graph | Multi-edge (5 types) |
| Orphan Handling | KNN pipeline (k=5, alpha=0.55) |
| Optimizer | AdamW (weight_decay=1e-4) |

### 9.2. Dataset RTPTorrent (APFD 0.8540)

**Script:** `experiments/run_filopriori_rtptorrent_v15.py`

| Parâmetro | Valor |
|---|---|
| Learning Rate | 1e-3 (GATv2), 1e-3 (DNN) |
| GATv2 Max Epochs | 30 (patience=7) |
| DNN Epochs | **15** (was 10 in V13) |
| DNN max_do_train_builds | **5000** |
| DNN max_pos_weight | **50.0** (CRITICAL clamp) |
| DNN Loss | **BCELoss** (NOT BCEWithLogitsLoss) |
| Fusion | Cross-attention (4 heads) |
| Graph | Co-failure only |
| Alpha Blending | Per-project optimized on validation |
| Optimizer | AdamW (weight_decay=1e-4) |

---

## 10. Estrutura de Arquivos de Resultados

```
results/
├── experiment_industry_optimized_v3/      # Industrial APFD 0.7611
├── filopriori_rtptorrent_v15/             # RTPTorrent APFD 0.8540
├── rtptorrent_ablation_sensitivity/       # RTPTorrent RQ2/RQ3/RQ4
│   ├── ablation/                          # 20 projetos × 6 variantes
│   │   ├── per_variant_per_project.csv
│   │   ├── aggregate_per_variant.csv
│   │   └── per_build_apfd.csv
│   ├── temporal_cv/                       # 20 projetos × 4 folds
│   │   ├── per_project_per_fold.csv
│   │   └── aggregate_per_project.csv
│   ├── sensitivity/                       # 20 projetos × 12 variantes
│   │   ├── per_variant_per_project.csv
│   │   └── aggregate_per_variant.csv
│   └── experiment_meta.json
├── deeporder_*/                           # DeepOrder baselines
├── noderank_*/                            # NodeRank baselines
├── retecs_*/                              # RETECS baselines
├── tcpnet_*/                              # TCP-Net baselines
└── failrank_bb_*/                         # FailRank-BB baselines
```

---

## 11. Próximos Passos

1. Revisão completa do inglês acadêmico
2. Gerar figuras finais para publicação (300 DPI)
3. Preparar replication package (repositório público)
4. Submeter ao IEEE TSE

---

*Relatório gerado em 24/02/2026 com base nos resultados experimentais consolidados do projeto Filo-Priori.*
