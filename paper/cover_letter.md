# Cover Letter — IEEE TSE Submission

**Date:** May 18, 2026

**To:** Editor-in-Chief
IEEE Transactions on Software Engineering (IEEE TSE)
IEEE Computer Society

**From:** Acauan C. Ribeiro (Corresponding Author), on behalf of all authors
Instituto de Computação (IComp), Universidade Federal do Amazonas (UFAM)
Manaus, AM, Brazil
acauan.ribeiro@icomp.ufam.edu.br

---

Dear Editor-in-Chief,

We are pleased to submit our manuscript entitled **"Filo-Priori: Co-Failure
Graph Attention for Test Case Prioritization in Continuous Integration"**
for consideration as a **Regular (Journal First)** publication in *IEEE
Transactions on Software Engineering*.

## Contribution Summary

Test Case Prioritization (TCP) in Continuous Integration (CI) pipelines is
critical for early fault detection in modern software development, yet
existing approaches typically treat test cases as independent entities,
ignoring the structural relationships that exist among them. We address
this gap with **Filo-Priori**, a graph-based deep learning framework that
models inter-test relationships through Graph Attention Networks (GATv2)
operating over a co-failure test relationship graph, complemented by a Deep
Neural Network (DNN) ensemble with data-driven alpha blending for
semantically-sparse settings.

Our contributions are fourfold:

1. **Architectural.** A graph-based pipeline for TCP centered on GATv2
   over a co-failure graph, with an optional DNN ensemble for
   metadata-sparse contexts and an optional multi-edge graph extension
   when richer metadata is available.

2. **Negative result on semantic features.** Generic Sentence-BERT
   embeddings provide no statistically significant improvement
   ($p = 0.309$) when structural features and co-failure propagation are
   present, confirmed by a Random-Fixed control experiment ($p = 0.965$).
   This challenges the common assumption that off-the-shelf semantic
   similarity is a strong predictor of test failure.

3. **Empirical.** Evaluation against five strong baselines (DeepOrder,
   TCP-Net, NodeRank, RETECS, FailRank-BB) on two complementary datasets:
   an industrial CI pipeline (52,102 executions, 1,339 builds) and the
   RTPTorrent benchmark (20 open-source Java projects, 2,937 builds with
   failures). Filo-Priori achieves APFD = 0.761 on the industrial dataset
   (significant gains over all five baselines, $p < 0.001$) and the highest
   numerical APFD of 0.854 on RTPTorrent. After Holm-Bonferroni correction,
   only gains over NodeRank and RETECS remain statistically robust,
   indicating that the framework's advantage scales with co-failure graph
   density.

4. **Insights.** Co-failure graph modeling is the decisive component
   ($+17.0\%$, $p < 0.001$); shallow GNN architectures (1 layer, 2 heads)
   suffice; and a simplified dual-balancing strategy (sampling + focal
   loss) prevents the mode collapse observed under triple-compensation.

## Originality and Exclusivity

We confirm that this manuscript reports **completely new research
results** that have not been previously published in any peer-reviewed
venue, in any form. No preliminary version, conference paper, workshop
paper, technical report, thesis chapter, or preprint of this work exists.
The manuscript is **not under consideration at any other journal,
conference, or workshop**, and will not be submitted elsewhere while under
review at IEEE TSE.

Given the original and complete nature of the contributions, we believe
this work fits the **Regular (Journal First)** category, and we have
included the required 200-word Novelty Statement as a separate file. We
understand that, if accepted, the manuscript will be eligible for
presentation under the ICSE Journal-First track at the discretion of
IEEE TSE and ICSE organizers.

## Reproducibility and Data Availability

A full replication package, including source code, trained models, training
configurations, and the anonymized industrial dataset, is publicly
available at:

**https://github.com/filotransformer/filopriori**

The RTPTorrent dataset used for cross-domain evaluation is publicly
available under CC BY 4.0 from the original authors.

## Ethics, Funding, and Conflicts of Interest

- **Funding:** This work was partially supported by the Coordenação de
  Aperfeiçoamento de Pessoal de Nível Superior (CAPES) — Finance Code 001,
  and by Motorola Mobility Comércio de Produtos Eletrônicos Ltda., under
  the auspices of the Brazilian Federal Law No. 8.387/1991.
- **Conflicts of Interest:** Author Jose Nascimento is employed by
  Motorola Mobility, which provided the industrial dataset analyzed in
  this study. The remaining authors declare no competing interests.
  Motorola Mobility had no role in study design, data analysis,
  interpretation of results, or the decision to submit this manuscript
  for publication. This is fully disclosed in the manuscript.
- **Generative AI Disclosure:** In accordance with IEEE policy, the use
  of AI assistance in manuscript preparation (limited to language
  polishing, LaTeX formatting suggestions, and documentation templates)
  is disclosed in the Acknowledgments section.

## Suggested Reviewers

We respectfully suggest the following reviewers based on expertise in
TCP, graph neural networks, and software testing. We have no personal or
professional relationship with any of them that could constitute a
conflict of interest:

1. **Prof. Mauro Pezzè** — Università della Svizzera italiana — TCP and
   regression testing expertise
2. **Prof. Tao Xie** — Peking University — software testing and ML for
   software engineering
3. **Prof. Antonia Bertolino** — ISTI-CNR — TCP and CI testing in
   industrial settings
4. **Prof. Lionel Briand** — University of Ottawa / SnT, Luxembourg —
   ML-based testing and empirical software engineering
5. **Prof. Sebastiano Panichella** — Zurich University of Applied
   Sciences — CI/CD testing and code review automation

## Excluded Reviewers

We respectfully request that the following parties not be assigned to
review this manuscript due to existing or recent collaborations:

- Researchers currently affiliated with UFAM (Federal University of
  Amazonas).
- Researchers currently or recently (within the past 24 months) employed
  by Motorola Mobility or affiliated entities.

## Closing

We thank you for considering our submission, and we look forward to the
review process. We are committed to addressing any feedback constructively
and within the journal's timeframes.

Sincerely,

**Acauan Cardoso Ribeiro**
(Corresponding Author)
On behalf of all co-authors

acauan.ribeiro@icomp.ufam.edu.br
Instituto de Computação (IComp)
Universidade Federal do Amazonas (UFAM)
Manaus, AM, Brazil
