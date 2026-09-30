# Paper (Bischoff et al. 2018)

> Source: `https://arxiv.org/abs/1809.03267`

---

**Feature Learning for Meta-Paths in Knowledge Graphs**

Sebastian Bischoff

**Abstract**

In this thesis, we study the problem of feature learning on heterogeneous knowledge graphs. These features can be used to perform tasks such as link prediction, classification and clustering on graphs. Knowledge graphs provide rich semantics encoded in the edge and node types. Meta-paths consist of these types and abstract paths in the graph. Until now, meta-paths can only be used as categorical features with high redundancy and are therefore unsuitable for machine learning models. We propose meta-path embeddings to solve this problem by learning semantical and compact vector representations of them. Current graph embedding methods only embed nodes and edge types and therefore miss semantics encoded in the combination of them. Our method embeds meta-paths using the skipgram model with an extension to deal with the redundancy and high amount of meta-paths in big knowledge graphs. We critically evaluate our embedding approach by predicting links on Wikidata. The experiments indicate that we learn a sensible embedding of the meta-paths but can improve it further.

---

- **arXiv ID**: `1809.03267`
- **Published**: 2018-09-07
- **Categories**: cs.LG, cs.SI, stat.ML
- **PDF**: https://arxiv.org/pdf/1809.03267
