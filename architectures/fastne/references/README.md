# References: FastNE

## Papers

- [`Fast_network_embedding_enhancement_via_high_order_proximity_Yang_Liu_Tu_2017.md`](papers/Fast_network_embedding_enhancement_via_high_order_proximity_Yang_Liu_Tu_2017.md) — primary source paper (from PaperVault `GNN/02-network-embedding/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **NEU** — https://github.com/thunlp/NEU — Official implementation of the NEU (Network Embedding Update) algorithm
- **THUNLP NRL collection** — https://github.com/thunlp — Tsinghua NLP group's network representation learning repository collection
- **Paper** — Published at IJCAI 2017; source code linked from the paper

## Related

**Predecessors:**
- Spectral Clustering (Tang & Liu, 2011) — Eigenvector computation on normalized Laplacian
- DeepWalk (Perozzi et al., 2014) — Skip-gram on random walks; low-order proximity
- LINE (Tang et al., 2015) — First/second-order proximity; explicitly low-order
- GraRep (Cao et al., 2015) — Explicit k-order proximity factorization; better but expensive
- TADW (Yang et al., 2015) — Matrix factorization with text; low-order proximity

**Successors:**
- NetMF (Qiu et al., 2018) — Unified matrix factorization framework connecting DeepWalk, LINE, node2vec
- GCN (Kipf & Welling, 2017) — Graph convolutional networks capture higher-order structure via layer stacking
- AROPE (Zhang et al., 2018) — Arbitrary-order proximity embedding via random matrix polynomial

**Alternatives:**
- node2vec (Grover & Leskovec, 2016) — Biased random walks; implicitly higher-order via walk length
- SDNE (Wang et al., 2016) — Deep autoencoder for network embedding
