# RETECS Rankings — Industrial QTA (277 builds)

Este pacote contém as **ordenações de testes** produzidas pelo baseline **RETECS**
(Network Agent + reward `tcfail`) para o dataset Industrial QTA, exatamente as mesmas
ordenações usadas para calcular o APFD reportado no paper (APFD médio = **0.6406**).

A partir desses rankings é possível calcular qualquer métrica de IR/ranking
(NDCG, MRR, MAP, Precision@k, etc.) sem precisar reexecutar o modelo.

## Como foram geradas

- Script: `experiments/run_retecs_industry_rankings.py`
- Configuração: idêntica ao experimento que produziu o APFD 0.6406
  - `agent_type = network`
  - `reward_func = tcfail`
  - `seed = 42`
  - Treino: `datasets/01_industry/train.csv` (todos os builds, em ordem)
  - Teste: `datasets/01_industry/test.csv` (apenas builds com >= 1 falha — 277 builds)
- O agente é treinado primeiro em todos os builds de treino, depois priorizado
  e atualizado (`update_history`) build a build no teste, mantendo o estado temporal
  consistente com o experimento original.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `rankings_per_build_industry.csv` | Long-format CSV com a ordenação produzida pelo RETECS para cada build de teste |
| `apfd_per_build_FULL_testcsv.csv` | (já existente) APFD por build do mesmo experimento — útil para sanity-check |

## Esquema do CSV `rankings_per_build_industry.csv`

| Coluna | Tipo | Descrição |
|---|---|---|
| `build_id` | string | Identificador do build (`Build_ID`) |
| `rank` | int | Posição 1-based no ranking (1 = primeiro priorizado pelo RETECS) |
| `test_id` | string | `TC_Key` do teste |
| `label` | int | 1 se o teste falhou neste build (relevante), 0 caso contrário |

Para cada build, as linhas estão ordenadas por `rank` crescente (1 → N).
O conjunto de testes em cada build é o universo do build (todos os testes executados
naquele build), e a soma de `label` por build é o número de falhas reais do build.

## Fórmulas das métricas

Para cada build *b*, com lista ordenada de testes $r_b = (t_1, t_2, \dots, t_N)$
e relevâncias binárias $y_i \in \{0, 1\}$ (1 = falhou):

### NDCG (Normalized Discounted Cumulative Gain)

$$
\text{DCG}_b = \sum_{i=1}^{N} \frac{2^{y_i} - 1}{\log_2(i + 1)}
\quad,\quad
\text{IDCG}_b = \sum_{i=1}^{F_b} \frac{1}{\log_2(i + 1)}
$$

$$
\text{NDCG}_b = \frac{\text{DCG}_b}{\text{IDCG}_b}
$$

onde $F_b = \sum_i y_i$ é o número de falhas no build *b*. O NDCG do dataset é a
média de $\text{NDCG}_b$ sobre os 277 builds.

> Observação: como $y_i \in \{0,1\}$, $2^{y_i} - 1 = y_i$, então
> $\text{DCG}_b = \sum_{i:\,y_i=1} 1/\log_2(i+1)$.

### MRR (Mean Reciprocal Rank)

Seja $\text{rank}_b^*$ a posição (1-based) do **primeiro** teste com $y_i = 1$ no
ranking do build *b*:

$$
\text{MRR} = \frac{1}{|B|} \sum_{b \in B} \frac{1}{\text{rank}_b^*}
$$

### MAP (Mean Average Precision)

Para o build *b*, a Average Precision (AP) é:

$$
\text{AP}_b = \frac{1}{F_b} \sum_{i=1}^{N} y_i \cdot \text{P@}i
\quad,\quad
\text{P@}i = \frac{1}{i} \sum_{j=1}^{i} y_j
$$

E o MAP é a média sobre os builds:

$$
\text{MAP} = \frac{1}{|B|} \sum_{b \in B} \text{AP}_b
$$

## Snippet de referência (Python / pandas)

```python
import numpy as np
import pandas as pd

df = pd.read_csv("rankings_per_build_industry.csv")
# Garante a ordem dentro do build
df = df.sort_values(["build_id", "rank"]).reset_index(drop=True)

def per_build_metrics(g):
    y = g["label"].to_numpy()
    n = len(y)
    F = int(y.sum())
    if F == 0:
        return pd.Series({"ndcg": np.nan, "mrr": np.nan, "ap": np.nan})

    # rank positions 1..n (already sorted by rank asc)
    pos = np.arange(1, n + 1)

    dcg  = (y / np.log2(pos + 1)).sum()
    idcg = (1.0 / np.log2(np.arange(1, F + 1) + 1)).sum()
    ndcg = dcg / idcg

    first_rel = pos[y == 1][0]
    mrr = 1.0 / first_rel

    cum_rel = np.cumsum(y)
    precisions_at_i = cum_rel / pos
    ap = (precisions_at_i[y == 1]).sum() / F

    return pd.Series({"ndcg": ndcg, "mrr": mrr, "ap": ap})

per_build = df.groupby("build_id").apply(per_build_metrics)
print("NDCG:", per_build["ndcg"].mean())
print("MRR :", per_build["mrr"].mean())
print("MAP :", per_build["ap"].mean())
```

## Notas

- Todos os 277 builds aqui têm pelo menos 1 falha; portanto NDCG, MRR e AP são
  sempre definidos (não há divisão por zero).
- O ranking do RETECS é uma permutação completa do conjunto de testes do build
  (sem cortes ou top-k). Métricas no estilo @k podem ser obtidas truncando o
  ranking em $k$.
- A semente (`seed=42`) e a sequência de treino/teste são as mesmas do experimento
  oficial; reexecutar o script reproduz exatamente o mesmo arquivo.
