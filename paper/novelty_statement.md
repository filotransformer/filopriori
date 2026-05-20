# Novelty Statement - Regular (Journal First) Submission

**Manuscript:** Filo-Priori: Co-Failure Graph Attention for Test Case Prioritization in Continuous Integration

**Authors:** Acauan C. Ribeiro, Eduardo L. Feitosa, Andre L. da Costa Carvalho, Eulanda M. dos Santos, Bruno F. Gadelha, Yan R. Soares, Jose Nascimento

**Target Venue:** IEEE Transactions on Software Engineering (IEEE TSE)

**Article Type:** Regular (Journal First)

---

## Statement of Novelty (200 words)

This manuscript reports completely new research results for Test Case
Prioritization (TCP) in Continuous Integration and is not a previously
published, posted, or submitted conference/workshop extension. No preliminary
version, technical report, thesis chapter, or preprint has been disseminated.
The submission is not under consideration elsewhere.

The novelty is substantive relative to IEEE TSE's Journal First criteria.
Filo-Priori introduces a TCP framework that treats tests as related entities
rather than independent instances, using GATv2 attention over a co-failure test
relationship graph and a DNN ensemble with validation-optimized alpha blending
for metadata-sparse settings. The study also contributes an explicit negative
result: generic Sentence-BERT embeddings do not significantly improve
prioritization when structural features and co-failure propagation are
available (p = 0.309), and a Random-Fixed control confirms that the apparent
semantic signal behaves as a stable identity proxy (p = 0.965). Empirically,
the work evaluates five strong baselines across an industrial CI pipeline
(52,102 executions, 1,339 builds) and RTPTorrent (20 Java projects, 2,937
failure builds). Finally, ablation, temporal validation, sensitivity analysis,
and per-edge-type experiments identify co-failure edges as the decisive graph
component and clarify the boundary conditions under which graph-based TCP
outperforms flat-feature models. These results are new, complete, and central
to the paper.

---

**Word count:** 200 words

**Confirmation of originality:** The authors confirm that the manuscript
contains new contributions to the field of Test Case Prioritization. The work
has not been published, posted as a preprint, submitted elsewhere, or derived
from a preliminary conference/workshop version. Should the work be accepted by
IEEE TSE, the authors understand that any ICSE Journal-First presentation
invitation would be handled under the journal's and conference's applicable
policies.

---

**Contact (Corresponding Author):** Acauan C. Ribeiro
(acauan.ribeiro@icomp.ufam.edu.br), Instituto de Computacao (IComp),
Universidade Federal do Amazonas (UFAM), Manaus, AM, Brazil.
