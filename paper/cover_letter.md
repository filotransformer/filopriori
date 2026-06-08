# Cover Letter - IEEE TSE Submission

**Date:** May 20, 2026

**To:** Editor-in-Chief  
IEEE Transactions on Software Engineering  
IEEE Computer Society

**From:** Acauan C. Ribeiro, Corresponding Author, on behalf of all authors  
Instituto de Computacao (IComp), Universidade Federal do Amazonas (UFAM)  
Manaus, AM, Brazil  
acauan.ribeiro@icomp.ufam.edu.br

---

Dear Editor-in-Chief,

On behalf of all authors, I am pleased to submit our manuscript entitled
**"Filo-Priori: Co-Failure Graph Attention for Test Case Prioritization in
Continuous Integration"** for consideration as a **Regular (Journal First)**
manuscript in *IEEE Transactions on Software Engineering*.

The manuscript addresses Test Case Prioritization (TCP) in Continuous
Integration (CI), a topic that fits TSE's interest in software testing,
software assessment methods, and empirical studies with potential impact on
software engineering practice. Existing TCP methods often score test cases as
independent entities. Filo-Priori instead models behavioral relationships
between tests using GATv2 attention over a co-failure test relationship graph,
complemented by a DNN ensemble with validation-optimized alpha blending for
metadata-sparse settings.

The main contributions are:

1. A graph-based TCP framework centered on co-failure relationships, with an
   optional multi-edge extension when richer CI metadata is available.
2. An empirical evaluation against five strong baselines on two complementary
   datasets: an industrial CI pipeline with 52,102 executions across 1,339
   builds and RTPTorrent with 20 open-source Java projects and 2,937 builds
   with failures.
3. Evidence that Filo-Priori achieves APFD = 0.761 on the industrial dataset
   with significant improvements over all five baselines, and the highest
   numerical APFD of 0.854 on RTPTorrent, where gains over NodeRank and RETECS
   remain robust after Holm-Bonferroni correction.
4. A negative result showing that generic Sentence-BERT embeddings do not
   provide a statistically significant improvement when structural features and
   co-failure propagation are available, supported by a Random-Fixed control
   experiment.
5. Ablation, temporal validation, and sensitivity analyses showing that
   co-failure graph modeling is the decisive component and clarifying when
   graph-based TCP is most beneficial.

This manuscript reports new research results that have not been previously
published in a peer-reviewed venue, posted as a preprint, submitted as a
conference/workshop paper, or disseminated as a technical report or thesis
chapter. It is not under consideration by any other journal, conference, or
workshop, and it will not be submitted elsewhere while under review by TSE. A
separate 200-word Journal First Novelty Statement is included with this
submission.

A public replication package, including the source code for Filo-Priori and
all baselines, trained model weights, training configurations, and scripts to
reproduce the RTPTorrent experiments end-to-end, is available at:

**https://github.com/filotransformer/filopriori**

The RTPTorrent dataset used for the cross-domain evaluation is publicly
available under CC BY 4.0 from its original authors. The industrial QTA
dataset is proprietary and governed by a non-disclosure agreement; it cannot
be redistributed, so the industrial experiments are not externally
reproducible. This is disclosed in the manuscript.

All authors have approved this submission. Funding, conflicts of interest, and
the use of generative AI assistance are disclosed in the manuscript. In
particular, authors Jose Carlos Rangel do Nascimento and Nícolas Riccieri
Gardin Assumpção are employed by Motorola Mobility, which provided the
industrial dataset and partially supported the research under the Brazilian
Federal Law No. 8.387/1991. Motorola Mobility had no role in the study
design, data analysis, interpretation of results, or decision to submit this
manuscript for publication.

If ScholarOne requests suggested or non-preferred reviewers, we will provide
them in the corresponding submission fields and will avoid individuals with
actual, perceived, or potential conflicts of interest.

Thank you for considering our manuscript for review in *IEEE Transactions on
Software Engineering*.

Sincerely,

**Acauan Cardoso Ribeiro**  
Corresponding Author, on behalf of all co-authors  
acauan.ribeiro@icomp.ufam.edu.br  
Instituto de Computacao (IComp)  
Universidade Federal do Amazonas (UFAM)  
Manaus, AM, Brazil
