# Paper (Milocco et al. 2025)

> Source: `https://arxiv.org/abs/2508.20706`

---

**Renormalizable Graph Embeddings For Multi-Scale Network Reconstruction**

Riccardo Milocco, Fabian Jansen, Diego Garlaschelli

**Abstract**

In machine learning, graph embedding algorithms seek low-dimensional representations of the input network data, thereby allowing for downstream tasks on compressed encodings. Recently, within the framework of network renormalization, multi-scale embeddings that remain consistent under an arbitrary aggregation of nodes onto block-nodes, and consequently under an arbitrary change of resolution of the input network data, have been proposed. Here we investigate such multi-scale graph embeddings in the modified context where the input network is not entirely observable, due to data limitations or privacy constraints. This situation is typical for financial and economic networks, where connections between individual banks or firms are hidden due to confidentiality, and one has to probabilistically reconstruct the underlying network from aggregate information. We first consider state-of-the-art network reconstruction techniques based on the maximum-entropy principle, which is designed to operate optimally at a fixed resolution level. We then discuss the limitations of these methods when they are used as graph embeddings to yield predictions across different resolution levels. Finally, we propose their natural 'renormalizable' counterparts derived from the distinct principle of scale invariance, yielding consistent graph embeddings for multi-scale network reconstruction. We illustrate these methods on national economic input-output networks and on international trade networks, which can be naturally represented at multiple levels of industrial and geographic resolution, respectively.

---

- **arXiv ID**: `2508.20706`
- **Published**: 2025-08-28
- **Categories**: physics.soc-ph, cond-mat.dis-nn, econ.GN
- **PDF**: https://arxiv.org/pdf/2508.20706
