# References: metapath2vec

## Papers

- [`metapath2vec_Scalable_Representation_Learning_for_Heterogene_Heterogeneous_2017.md`](papers/metapath2vec_Scalable_Representation_Learning_for_Heterogene_Heterogeneous_2017.md) — primary source paper (from PaperVault `GNN/02-network-embedding/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **Code:** https://github.com/yangji9181/metapath2vec
- **arXiv:** https://arxiv.org/abs/1707.03140
- **DOI:** https://doi.org/10.1145/3097983.3098036

## Related

- **DeepWalk** — Predecessor; homogeneous random walk + skip-gram. metapath2vec extends to heterogeneous networks via meta-path-guided walks.
- **node2vec** — Predecessor; flexible biased walks for homogeneous networks. metapath2vec adapts the walk concept to heterogeneous settings.
- **LINE** — Predecessor; 1st and 2nd order proximity for homogeneous networks.
- **PTE** — Related; predictive text embedding for semi-supervised heterogeneous text networks. metapath2vec is unsupervised and outputs multi-type embeddings.
- **PathSim** — Predecessor; meta-path-based similarity search. metapath2vec learns embeddings that subsume meta-path-based similarity without requiring connected meta-paths.
- **HIN2Vec** — Concurrent work; jointly learns node and relationship vectors via multi-task binary classification for HINs.
- **HERec** — Successor; heterogeneous network embedding with recommender systems focus.
- **HGT** — Successor; Heterogeneous Graph Transformer using type-specific attention for heterogeneous graphs.
