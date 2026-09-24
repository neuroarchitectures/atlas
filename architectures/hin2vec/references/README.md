# References: HIN2Vec

## Papers

- [`HIN2Vec_Explore_Meta_paths_in_Heterogeneous_Information_Netw_Fu_Lei_2017.md`](papers/HIN2Vec_Explore_Meta_paths_in_Heterogeneous_Information_Netw_Fu_Lei_2017.md) — primary source paper (from PaperVault `GNN/02-network-embedding/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **arXiv:** https://arxiv.org/abs/1711.08338
- **DOI:** https://doi.org/10.1145/3132847.3132953

## Related

- **DeepWalk** — Predecessor; homogeneous network embedding. HIN2Vec extends to heterogeneous networks with meta-path exploration.
- **LINE** — Predecessor; 1st/2nd order proximity for homogeneous networks. HIN2Vec generalizes to multi-relational HINs.
- **node2vec** — Predecessor; flexible homogeneous walks. HIN2Vec constrains walks to meta-paths for semantic relevance.
- **PTE** — Predecessor; predictive text embedding for semi-supervised heterogeneous text networks. HIN2Vec is unsupervised and outputs multi-type embeddings.
- **ESim** — Predecessor; embedding for short texts with meta-path guidance. HIN2Vec generalizes to HINs with joint relationship learning.
- **HINE** — Predecessor; heterogeneous information network embedding. HIN2Vec improves by jointly learning relationship vectors.
- **metapath2vec / metapath2vec++** — Concurrent work; meta-path-based random walks with heterogeneous skip-gram. Similar ideas but different model architecture (skip-gram vs. neural classifier).
- **HERec** — Successor; heterogeneous network embedding for recommendation systems.
- **TransE** — Related; knowledge graph embedding using translation operations. HIN2Vec's Hadamard product is a complementary combination strategy for multi-relational data.
