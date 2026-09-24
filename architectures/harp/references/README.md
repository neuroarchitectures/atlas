# References: HARP

## Papers

- [`HARP_Hierarchical_Representation_Learning_for_Networks_Chen_Perozzi_Hu_etal_2017.md`](papers/HARP_Hierarchical_Representation_Learning_for_Networks_Chen_Perozzi_Hu_etal_2017.md) — primary source paper (from PaperVault `GNN/02-network-embedding/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **Code:** https://github.com/GTmac/HARP
- **arXiv:** https://arxiv.org/abs/1706.07845

## Related

- **DeepWalk** — Base method wrapped by HARP; random walk + skip-gram. HARP(DW) improves it via hierarchical initialization.
- **LINE** — Base method wrapped by HARP; 1st/2nd order proximity. HARP(LINE) recovers global structure (rings, grids) that LINE misses.
- **node2vec** — Base method wrapped by HARP; biased random walks. HARP(N2V) improves it on classification tasks.
- **GraRep** — Related; captures global structure via k-step transition matrices. HARP addresses the same global-structure problem via coarsening.
- **GraphGAN** — Related; adversarial graph embedding. HARP's meta-strategy could wrap GraphGAN for improved initialization.
- **struc2vec** — Related; structural identity embedding. Complementary approach; HARP could potentially wrap struc2vec.
- **Multigrid methods (numerical linear algebra)** — Conceptual predecessor; coarse-to-fine paradigm for solving PDEs. HARP adapts this idea to graph embedding.
- **Cluster-GCN** — Successor; clustering-based graph coarsening for scalable GNN training. Extends the multi-level idea to deep GNNs.
- **GraphSAINT** — Successor; graph sampling for scalable GNN training. Different approach to graph-level scalability.
