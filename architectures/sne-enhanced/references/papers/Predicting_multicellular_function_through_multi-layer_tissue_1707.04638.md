# Predicting multicellular function through multi-layer tissue networks

> Source: `https://arxiv.org/abs/1707.04638`

---

**Authors:** Marinka Žitnik, Jure Leskovec

**Published:** 2017

**Concepts:** Computer science, Hierarchy, Multicellular organism, Function (biology), Computational biology

## Abstract

MOTIVATION: Understanding functions of proteins in specific human tissues is essential for insights into disease diagnostics and therapeutics, yet prediction of tissue-specific cellular function remains a critical challenge for biomedicine. RESULTS: Here, we present OhmNet , a hierarchy-aware unsupervised node feature learning approach for multi-layer networks. We build a multi-layer network, where each layer represents molecular interactions in a different human tissue. OhmNet then automatically learns a mapping of proteins, represented as nodes, to a neural embedding-based low-dimensional space of features. OhmNet encourages sharing of similar features among proteins with similar network neighborhoods and among proteins activated in similar tissues. The algorithm generalizes prior work, which generally ignores relationships between tissues, by modeling tissue organization with a rich multiscale tissue hierarchy. We use OhmNet to study multicellular function in a multi-layer protein interaction network of 107 human tissues. In 48 tissues with known tissue-specific cellular functions, OhmNet provides more accurate predictions of cellular function than alternative approaches, and also generates more accurate hypotheses about tissue-specific protein actions. We show that taking into account the tissue hierarchy leads to improved predictive power. Remarkably, we also demonstrate that it is possible to leverage the tissue hierarchy in order to effectively transfer cellular functions to a functionally uncharacterized tissue. Overall, OhmNet moves from flat networks to multiscale models able to predict a range of phenotypes spanning cellular subsystems. AVAILABILITY AND IMPLEMENTATION: Source code and datasets are available at http://snap.stanford.edu/ohmnet . CONTACT: jure@cs.stanford.edu.

**PDF:** https://arxiv.org/pdf/1707.04638